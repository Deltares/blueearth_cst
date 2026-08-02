# P3-2a — Model-Independent Climate Analysis: Design (ACCEPTED)

> **Status: ACCEPTED** — approved at human gate G2 (2026-07-24) under
> arbitration authority. Final revision (v4) followed the user's binding
> arbitration (2026-07-24, final) of external review round 2
> (the review record; major ext2-1/ext2-2, minor ext2-3 — all accepted with
> arbitrated fix variants), which followed external round 1
> (the review record) and the internal lens panel (risk / architecture /
> repo-fit; blocking arch-1). G1 framing (problem, constraints, decision
> criteria, provisional selected alternative) was approved 2026-07-24; this
> revision is strictly confined to the three arbitrated findings and changes
> design detail only, not the approved framing.
> Per-finding disposition in the review record (`climate-analysis-design-review-record.md`); what changed in §11 (v2, v3, v4).
> **Milestone:** P3-2a (functional decomposition of climate analysis + wf1
> subcatchment-plot re-source). Absorbs the R6-deferred functional decomposition
> (`dev/milestones/r06/structural-refactor-design.md` §8).
> **Genre:** decision-record (milestone/refactor). **Commit prefix:** `p32a:`.
> **Author role:** cst-architect. **Run:** `p32a-climate-analysis`.
> **Scope authority:** `dev/milestones/p32a/climate-analysis-intake.md`
> (its four "Confirmed scoping decisions" are fixed anchors, not reopened here).
> This doc is self-contained: a reviewer needs only this file plus the cited paths.

---

## 1. Problem statement

Climate analysis in this repo is entangled with the hydrological model build in a
way that blocks the standing modularization direction
(`modularization-climate-analysis-subworkflow` memory) and leaves the
"model-independent climate analysis" aspiration unrealized in code.

Two concrete entanglements, both grounded in the tree:

- **Structural.** The climate-analysis helpers are scattered across
  workflow-specific submodules with no shared home. `extract_historical_climate.py`
  (raw gridded extraction; depends only on `region.geojson` + catalog + window)
  lives under `experiment/` because wf3 is its sole current consumer
  (`blueearth_cst/experiment/extract_historical_climate.py`;
  `Snakefile_climate_experiment:194` rule 3.02). `prepare_climate_data_catalog.py`
  lives under `projections/`. The subcatchment aggregation
  (`climate_forcing_by_subcatchment`) lives under `model/`
  (`blueearth_cst/model/climate_forcing.py`). The plotting primitives (`plot_clim`,
  `plot_map`) are already in `shared/` — the only zero-move pieces. R6 deliberately
  placed the model-independent helpers to keep them *liftable* but explicitly
  deferred the lift (`dev/milestones/r06/structural-refactor-design.md` §8, "DEFER"): a later
  milestone owns the *behavioral* re-source, "which is where a computational-path
  change belongs."

- **Behavioral (the ADR-0002 coupling).** The wf1 subcatchment climate plots
  (`plot_results.py` §4) source their P/T/EP series from `mod.forcing.data` — the
  ERA5-derived `inmaps_historical.nc` forcing *already loaded into the built Wflow
  model* — reduced per subcatchment via `mod.staticmaps.data["subcatchment"]`
  (`blueearth_cst/model/plot_results.py:161-165`; ADR 0002, accepted 2026-07-21).
  That is the *model's regridded forcing* (native catalog grid → model grid), so
  the plots cannot run without a fully-built, forced model. A genuinely
  model-independent climate analysis would source **raw gridded climate** (region +
  catalog + window), decoupled from the build. ADR 0002 itself records the coupling
  as the pragmatic choice available at the time and cross-references the
  forthcoming decomposition; `climate_forcing.py:1-7` was written "separate from the
  plotting/model code so the same aggregation can back a model-independent
  climate-analysis component later."

The cost of leaving this: climate QA/plots stay chained to the model build (cannot
run a climate screen without building Wflow), the model-independent-subworkflow
direction and the P3-2b model-swap contracts have no settled climate-analysis home
to build on, and "model-independent" remains aspirational rather than checkable.

## 2. Goals / Non-goals

### Goals

1. **G1 — a `blueearth_cst/climate_analysis/` subpackage** whose public functions
   are strictly **model-independent**: they take region/catalog/window and plain
   gridded/vector data (xarray, GeoDataFrame, paths), and **never import or receive
   a `WflowSbmModel`**. Model-independence is a checkable property (§4 criterion C1),
   not a slogan.
2. **G2 — re-source the wf1 subcatchment climate plots from raw gridded climate**
   (region + catalog + window), unwinding the ADR-0002 `mod.forcing.data` coupling.
   This is the milestone's **single sanctioned value change**, accepted via visual
   QA + characterized diff (intake decision 4).
3. **G3 — mechanically rewire wf2/wf3** to consume the lifted modules, value-identical
   (semantic-tree-diff clean modulo the documented wf1 plot change), preserving the
   P3-1 keyed store + `.guard_ok` wiring **exactly**.
4. **G4 — reproducible, per-commit-runnable migration**: every commit leaves the
   repo `--dry-run`-clean and `pytest tests/test_cli.py`-green; the value change is
   recorded (characterized diff + manifest re-record scope) so the run re-runs from
   recorded state.

### Non-goals

- **No 4th Snakefile / no new entry point** (intake decision 3). The three
  `Snakefile_*` entry points, the `run_workflows.py` wrapper contract, config
  `workflows:` sections, and the GUI-facing platform surface are unchanged. A
  standalone climate-screening entry point is a cheap later addition once the
  subpackage exists.
- **No P3-2b interchange contracts** (model-swap netCDF/state shapes) — separate
  cycle.
- **No new plot types or analysis products** — re-source and lift only.
- **No realization/stress-test file-format redesign** — parked (P3-3 candidate).
- **No change to any value outside the named wf1 plot re-source** — everything else
  stays value-identical.
- **No re-architecture of the P3-1 keyed store / drift-guard contract** — it is
  preserved verbatim, not extended to feed wf1 (see §4.2 / §6.2).

## 3. Constraints (standing; restated for this milestone)

- **CST automation scope.** hydromt / hydromt_wflow / wflow conventions are consumed
  verbatim; no upstream re-engineering, no patching the vendored packages (AGENTS.md
  Hard Constraints; `stay-within-cst-automation-scope` memory). The lifted extraction
  uses `hydromt.DataCatalog` + `get_rasterdataset` exactly as today.
- **No new dependencies** without prior user approval (`new-dependency-requires-approval`
  memory). The lift and re-source reuse existing packages (xarray, hydromt,
  geopandas, matplotlib) only.
- **`blueearth_cst/weathergen/*.R` content untouched.**
- **Value-identical discipline** for everything except the named wf1 plot re-source:
  semantic-tree-diff + baseline-manifest discipline (R3/R5 style). The re-source is
  the one sanctioned value change (intake decision 4).
- **Platform surface unchanged.** Three Snakefile entry points, `run_workflows.py`
  wrapper contract, config `workflows:` sections, GUI-facing surface — all unchanged.
- **Naming** per `dev/conventions/naming.md`: snake_case modules, lowercase acronyms,
  `_path`/`_dir` vs `_ds`/`_df`/`_gdf`/`_cfg`. Existing module filenames are
  grandfathered; a moved file keeps its name. Renaming a §7 contract surface needs a
  migration note.
- **Every move commit stays runnable** (decision-record reference, migration rule):
  no bare `git mv` that leaves the tree un-runnable between commits — either an atomic
  move+reference-rewrite commit justified as one mechanical transform, or a transitional
  re-export shim.

## 4. Decision criteria

A design choice below is judged against, in priority order:

- **C1 — Model-independence is checkable.** No `climate_analysis/` public function
  imports or is passed a `WflowSbmModel`. Operationally: `grep` for `WflowSbmModel`,
  `mod.forcing`, `mod.staticmaps` in `blueearth_cst/climate_analysis/**` returns
  nothing; any model-specific loading (e.g. reading the subcatchment raster from
  `staticmaps.nc`) lives in a `model/`-side caller, not in the subpackage.
- **C2 — Value-identity preserved off the sanctioned change.** wf2/wf3 outputs and
  the wf1 non-climate-plot outputs are byte/semantically identical pre/post; only the
  named wf1 subcatchment climate plots change, and that change is characterized and
  accepted.
- **C3 — P3-1 contract preserved exactly.** The dataset+window keyed store
  (`climate_historical/<key>/`), the `.guard_ok` key-level artifact consumed
  `ancient()` by `extract_climate_grid` only, the per-experiment sentinel, and the
  input-set-invariance property (`dev/milestones/p31/experiment-structure-design.md` §3a/§3d/§4)
  are untouched.
- **C4 — Pipeline order preserved.** wf1 runs before wf3 (AGENTS.md; `run_workflows.py`
  model→projections→experiment). No wf1 rule may depend on a wf3-produced artifact.
- **C5 — Per-commit runnability & reproducibility.** Each commit `--dry-run`-clean +
  `test_cli.py`-green; the value change re-runs from recorded state.
- **C6 — Minimal churn, forward-compatible.** Move only what the decomposition
  requires; keep the layout ready for the deferred standalone entry point and P3-2b
  without pre-building unrequested structure.

## 5. Selected approach

### 5.1 Subpackage layout and public surface

Create `blueearth_cst/climate_analysis/` and lift the three model-independent
climate helpers into it. The plotting primitives stay in `shared/` (already
model-independent, consumed by multiple stages — moving them would be churn without
a decoupling win).

```
blueearth_cst/
  climate_analysis/
    __init__.py                       # empty (package marker, matches repo convention)
    extract_historical_climate.py     # MOVED from experiment/ — prep_historical_climate(region, catalog, window)
    subcatchment_climate.py           # MOVED from model/climate_forcing.py — zone-aggregation, renamed (see below)
    prepare_climate_data_catalog.py   # MOVED from projections/
  model/
    extract_climate_wf1.py            # NEW — wf1 wrapper script for the extraction rule (ext1-4, §5.2)
    climate_parity.py                 # NEW — model-parity transform of raw climate (§5.2)
  shared/
    func_plot_signature.py            # UNCHANGED — plot_clim stays here (zero-move)
    plot_map.py                       # UNCHANGED (zero-move)
```

**Two NEW `model/`-side modules (ext1-2, ext1-4).** Both consume model artifacts
(`staticmaps.nc`) or model-parity semantics, so per the C1 boundary they live in
`model/`, not in the subpackage:

- `model/extract_climate_wf1.py` — the wf1 extraction rule's `script:` target. A
  thin wrapper that imports `prep_historical_climate` from
  `blueearth_cst.climate_analysis.extract_historical_climate`, reads
  `sm.input.basin_nc` (`staticmaps.nc`), derives the extraction bbox from it
  (derivation + tolerance in §5.2, ext1-5), and calls the shared function with
  `bbox=`. On the chirps/chirps_global branch it then relocates the extraction's
  orography sidecar to the rule's declared `oro_nc` output (ext2-1, §5.2); it also
  carries the belt-and-braces eobs assert behind the parse-time guard (ext2-3,
  §5.2). The moved module's own wf3 `__main__` block stays **verbatim** and is
  exercised only by wf3 rule 3.02 (ext1-4; full contract in §5.3).
- `model/climate_parity.py` — `model_parity_climate(ds_raw, dem_model,
  dem_forcing, pet_method) -> xr.Dataset` reproducing the forcing build's
  regrid + corrections + PET on the raw extraction, via the exact hydromt
  callables named in §5.2. Called from `plot_results.py` §4 (in-process at plot
  time; no new persisted production intermediate — the QA ladder in §5.5
  persists its own states via a dev-side script). Takes plain xarray objects
  only; the model-artifact loading (`staticmaps.nc["land_elevation"]`,
  `["subcatchment"]`) stays in `plot_results.py`.

**Which modules move vs stay.**

| Module (current path) | Action | New path | Rationale |
| --- | --- | --- | --- |
| `experiment/extract_historical_climate.py` | move | `climate_analysis/extract_historical_climate.py` | Raw gridded extraction; already `(region, catalog, window)`-shaped (`prep_historical_climate` signature, lines 48-56). Sole consumer today is wf3 rule 3.02; wf1 becomes a second consumer (§5.2). |
| `model/climate_forcing.py` | move (+ optional rename, OQ-1) | `climate_analysis/subcatchment_climate.py` | Zone-aggregation of gridded climate; already model-independent (takes `forcing`, `subcatchment` DataArrays — `climate_forcing_by_subcatchment`, lines 17-58; author note lines 1-7). Filename `climate_forcing.py` is model-flavored; a rename to `subcatchment_climate.py` is a taste call (OQ-1). **The function signature and raster-groupby algorithm are unchanged** under the selected Option B (§6.3). |
| `projections/prepare_climate_data_catalog.py` | move | `climate_analysis/prepare_climate_data_catalog.py` | Catalog-prep helper R6 identified as a mechanical later move; a wf3 consumer (chirps catalog prep) rewires. |
| `shared/func_plot_signature.py` (`plot_clim`) | stay | — | Already `shared/`, model-independent, multi-consumer. Zero-move. |
| `shared/plot_map.py` | stay | — | Same. |

**Module rename detail (naming.md), OQ-1.** `climate_forcing.py` →
`subcatchment_climate.py` is a *cosmetic* rename (the new home is climate-analysis and
"forcing" names the model's regridded input specifically). Under Option B the function's
signature and algorithm are unchanged, so the rename is pure taste/churn tradeoff — kept
as **OQ-1**, recommend rename. The function name is likewise a taste call: keep
`climate_forcing_by_subcatchment` (grandfathered, lowest churn) or rename to
`climate_by_zone` for the new home. Recommend keeping the existing function name to
minimise churn and keep the one importer + `test_climate_forcing.py` byte-stable, since
Option B does not otherwise touch either.

**Model-independent signature set (the public surface).**

- `prep_historical_climate(region_fn, fn_out, data_libs, clim_source, *, starttime,
  endtime[, bbox=None])` — **signature-compatible** with the verbatim wf3 call: the
  optional keyword-only `bbox=` (arch-1 fix (b), §5.2) lets the wf1 caller pass a bbox
  derived from `staticmaps.nc` while wf3 keeps calling with `region_fn` only, so wf3's
  output is byte-identical. Takes region path/bbox + catalog + window; writes a netCDF.
  C1-compliant.
- `climate_forcing_by_subcatchment(forcing, subcatchment)` — **unchanged signature and
  raster-`groupby` algorithm** (`climate_forcing.py:17-58`). Spatial-mean a gridded
  climate dataset (`precip`/`temp`/`pet`) per subcatchment id → `(index, time)` dataset
  with `P/T/EP_subcatchment` keys `plot_clim` expects. Takes plain xarray objects
  (`forcing` Dataset + `subcatchment` DataArray on a **shared grid**); **never a
  model**. Because Option B (§6.3) reprojects the raw climate onto the **model grid**
  before calling this, the shared-grid precondition holds exactly as in the old path, so
  the `FORCING_TO_CLIM` mapping, the `_FillValue` nodata handling, and the `(index,
  time)` output contract are all preserved. **`tests/test_climate_forcing.py` (a second
  live caller of this function — repo-1) keeps passing verbatim**, since neither the
  signature nor the reduction changes.
- `prepare_clim_data_catalog(fns, data_libs_like, source_like, fn_out, oro_path)` —
  **unchanged signature**, verbatim from `prepare_climate_data_catalog.py:9`.

**The one coupling and where it lives (C1 resolution).** Per-subcatchment aggregation
*needs* the subcatchment zones, which are neither region nor catalog. The
model-independence criterion is interpreted as **"takes no `WflowSbmModel` object"**, not
"takes only region/catalog/window scalars." The subcatchment zone **raster** is read from
the wf1 built model's `staticmaps.nc["subcatchment"]` **by the wf1-side caller** (in
`model/`, §5.2) and passed to `climate_forcing_by_subcatchment(forcing, subcatchment)` as
a plain **xarray DataArray** on the model grid. The subpackage stays model-clean (no model
import, only xarray); the model-specific loading stays in wf1. **Scope note (risk-2):**
this makes the *extraction* (`prep_historical_climate`) genuinely runnable without a built
model, but the *aggregation* still depends on the built model for its subcatchment-zone
raster via the wf1-side caller — §7 states this honestly and does not claim aggregation is
model-free.

### 5.2 wf1 re-source mechanism (raw gridded climate, no `mod.forcing.data`)

**What changes at the plot site.** `plot_results.py` §4 currently builds `ds_clim`
from `mod.forcing.data` + `mod.staticmaps.data["subcatchment"]`
(`plot_results.py:161-165`). After the change it builds `ds_clim` from **raw gridded
climate extracted for the basin region over the historical window**, aggregated by
the subcatchment zones (still read from `staticmaps.nc`, which is a legitimate wf1
artifact — the *model build* output, not the model's regridded forcing).

**Where extraction happens in the wf1 DAG — a NEW wf1-owned extraction rule.** The
wf1 plots get raw gridded climate from a **wf1-owned extraction**, reusing the lifted
`prep_historical_climate` function, writing to a **wf1-owned path** distinct from both
the P3-1 keyed store and the existing wf1 forcing:

- New rule `1.NN extract_climate_grid_wf1` (numbering resolved at insert; §8):
  - `input:` `basin_nc = ancient({basin_dir}/staticmaps.nc)` — a **declared rule-1.03
    output** (`Snakefile_model_creation:102`), so it has a real producer edge in wf1
    (see arch-1 resolution below). The extraction reads the basin bbox from
    `staticmaps.nc` (the model-build output, not the model's regridded forcing) rather
    than from `region.geojson`.
  - `params:` `data_sources = DATA_SOURCES`, `clim_source` (from
    `shared.clim_historical`), `starttime`/`endtime` (from `shared.historical_window`)
    — the same three params wf1 already reads for `setup_runtime`
    (`Snakefile_model_creation:169-171`).
  - `output:` `climate_nc = {project_dir}/climate_historical/wf1_raw/extract_historical.nc`
    (**wf1-owned; distinct** from `climate_historical/<key>/` (wf3 store) and from
    `climate_historical/wflow_data/inmaps_historical.nc` (wf1 forcing)). **On the
    chirps/chirps_global branch the rule additionally declares** `oro_nc =
    {project_dir}/climate_historical/wf1_raw/orography.nc` — the extraction's
    orography sidecar as an **explicit rule output with a stable,
    `clim_source`-independent filename** (ext2-1). The branch is resolved at
    DAG-parse time from the config's `clim_historical` (the same value the rule
    already receives as its `clim_source` param), so the output dict is
    `{climate_nc}` on era5 and `{climate_nc, oro_nc}` on chirps/chirps_global — no
    dynamic outputs. The shared function writes the sidecar as
    `{clim_source}_orography.nc` beside `fn_out`
    (`extract_historical_climate.py:143-144` — an incidental dirname-based
    filename construction); the wf1 wrapper **moves that file to the declared
    `sm.output.oro_nc` path after the call**, so the shared module stays verbatim
    while the declared edge — not a filename convention — carries the contract.
    Snakemake then enforces existence and co-provenance: on an incremental or
    partially cleaned run a missing or stale sidecar re-runs the extraction rule
    as a whole, so `climate_nc` and `oro_nc` can never come from different
    extractions.
  - `script:` `blueearth_cst/model/extract_climate_wf1.py` — the **dedicated wf1
    wrapper** (ext1-4, §5.1/§5.3), NOT the moved shared module: the moved module's
    `__main__` reads `sm.input.prj_region`, which this rule does not declare;
    pointing the wf1 rule at it would fail at execution. The wrapper owns the
    wf1-shaped Snakemake dispatch; the moved module's wf3 `script:` block stays
    verbatim.
- `plot_results` rule 1.10 gains this raw netCDF as an `input:` and **drops the
  `mod.forcing.data` dependency**. Its current `forcing_path` input
  (`inmaps_historical.nc`, `Snakefile_model_creation:216`) is **removed** — the plots
  no longer read the model forcing. **The orography source for the parity module is
  a branch-resolved rule-1.10 input, never sibling-file discovery (ext2-1):** on the
  chirps/chirps_global branch rule 1.10 declares
  `input: oro_nc = {project_dir}/climate_historical/wf1_raw/orography.nc` — the
  extraction rule's declared sidecar output above, giving the correction-critical
  DEM a real producer→consumer edge — and `plot_results.py` reads it as
  `sm.input.oro_nc` (it must **not** derive the path from `climate_nc`'s dirname);
  on the era5 branch no sidecar input exists and `dem_forcing` is fetched from the
  catalog (`era5_orography`) via the `data_sources`/`clim_source` params, as below.
  Both rules resolve the branch at DAG-parse time from the same config key, so the
  producer's output dict and the consumer's input dict cannot disagree. The
  `script = "blueearth_cst/model/plot_results.py"`
  **input label** (`Snakefile_model_creation:217`, a second declaration of the script
  path alongside the `script:` directive at :228) is **left intact** — `plot_results.py`
  is not moved (repo-5). The other §4-adjacent reads — discharge from `output.csv`,
  subcatchment raster from `staticmaps.nc` via the model root — stay.

**arch-1 resolution — the extraction input needs a declared producer edge.** The v1
draft declared `input: ancient({basin_dir}/staticgeoms/region.geojson)`. Verified
against all three Snakefiles: **`region.geojson` has no producer rule** — it is an
undeclared side-effect of rule 1.03's `hydromt build wflow_sbm` shell body, and appears
only as an `ancient(...)` **input** in wf2/wf3 (`Snakefile_climate_projections:90,114`;
`Snakefile_climate_experiment:197`), never as an `output:`. `ancient()` suppresses the
timestamp check, not the existence/producer requirement. wf3 tolerates this only because
it is a **separate, later** invocation where wf1 has already written the file to disk;
a wf1 rule consuming `region.geojson` **intra-workflow** has no such safety — on a fresh
production `project_dir` the file does not exist at DAG-build time and nothing produces
it, so `snakemake all -s Snakefile_model_creation` raises `MissingInputException` at DAG
construction. This is the **same C4 producer/order failure class the design uses to
reject store-reuse** (§6.2); reintroducing it inside wf1 would be inconsistent, and it
would also red the commit-3 gate — `tests/test_cli.py::test_snakefile_cli_model_creation`
dry-runs wf1 on the **default tracked config**, which does **not** stage `region.geojson`
(only `config_with_staged_region`, used by the wf2/wf3 dry-run tests, stages it).

**Fix (selected): source the extraction bbox from `staticmaps.nc`.** The new rule takes
`basin_nc = ancient({basin_dir}/staticmaps.nc)` — a genuine rule-1.03 `output:`
(`:102`), giving the extraction a declared producer edge inside wf1. This keeps every
per-commit gate green on the default config (the dry-run resolves the DAG because
`staticmaps.nc` has rule `create_model` as producer) and needs **no** new region-preview
rule. The `prep_historical_climate` function currently reads the bbox from a region
geojson (`extract_historical_climate.py:79,93` — `region.geometry.total_bounds`); the
wf1 wrapper (`model/extract_climate_wf1.py`, §5.1) derives the same bbox from
`staticmaps.nc` and passes it in. **Selected (was "preferred" in v2 — now pinned):**
add an optional keyword-only `bbox=None` parameter to `prep_historical_climate`
(and relax `region_fn` to `Optional`, exactly one of the two required); the wrapper
passes the `staticmaps.nc`-derived bbox while wf3 keeps passing `region_fn` — a
**signature-compatible** addition (wf3 call unchanged, so wf3's
`extract_historical.nc` is byte-identical). This makes the producer edge and the
bbox source the *same* declared artifact, closing arch-1 cleanly without relying on
an undeclared file. The declared Snakemake input is `staticmaps.nc`, so the DAG
builds on a fresh project and every dry-run gate passes. (The v2 fallback (a) —
passing the on-disk-but-undeclared `region.geojson` path — is dropped: the bbox
recovery below is trivially clean, so the fallback's undeclared-file smell buys
nothing.) **This is an implementation-detail change within the G1-approved framing**
— the source is still "raw climate independent of the model's stored forcing
artifact"; only the *declared producer edge* moves from a non-existent-producer file
to a real one.

**Bbox derivation and its relation to the region-derived extent (ext1-5).** The
wrapper opens `staticmaps.nc` with xarray + the hydromt raster accessor and takes
`bbox = ds_sm.raster.bounds` — the `(xmin, ymin, xmax, ymax)` bounds of the model
grid, the same tuple form `region.geometry.total_bounds` yields on the wf3 path.
The two extents are close but not identical: rule 1.03's `setup_basemaps` builds
the model grid to cover the basin geometry snapped outward to `model_resolution`
(default `0.00833°`, `Snakefile_model_creation:35`), so each edge of the staticmaps
bbox can sit up to ~one model cell outside the region's tight bounds (plus grid-
origin alignment). **Assertion (unit test, commit 3):** on the test fixture, derive
both bboxes and assert per-edge `|bbox_staticmaps − bbox_region| ≤ 2 ×
model_resolution` (≈ 0.017°); record the actual per-edge offsets in the baseline
note. **Why the difference cannot change the extraction:** `prep_historical_climate`
passes `buffer=1` to `get_rasterdataset` — one *source-dataset* cell (≥ 0.05°
chirps, 0.25° era5), an order of magnitude larger than the ≤ 2-model-cell bbox
shift — so the selected source cells are identical except at exact source-cell-edge
ties, where the buffer still guarantees coverage and interior `nearest_index`
reprojection is unchanged. **Empirical closure (commit 5 QA):** on the fixture,
wf1's `wf1_raw/extract_historical.nc` (staticmaps-bbox) is compared `allclose`
against wf3's keyed-store `extract_historical.nc` (region-bbox; same `clim_source`
+ window on the fixture) — both are outputs of the *same* function, so equality
proves the bbox swap changed nothing.

**Why a separate wf1 extraction and NOT reuse of the P3-1 keyed store (C3, C4).**
Reusing `climate_historical/<key>/extract_historical.nc` from wf1 is **blocked by
pipeline order**, not merely inelegant:

- wf1 runs **before** wf3 (AGENTS.md; `run_workflows.py` fixed order). In a fresh
  project the keyed store does not exist at wf1 time, so a wf1 `input:` on it raises
  `MissingInputException` (C4 violation). The store is produced by a **wf3** rule
  (3.02) that is *guard-gated* by the P3-1 drift guard (`.guard_ok`), so a wf1
  dependency would invert the documented order and couple wf1 to a downstream,
  guard-gated artifact.
- The shared `clim_source + historical_window` key is a **red herring** for reuse:
  wf1 and wf3 do resolve the same dataset+window under the R01 schema, but a shared
  *key* does not help when the *producer* is downstream. Making wf1 the producer and
  wf3 the consumer would re-architect the P3-1 `.guard_ok` / `extract_climate_grid` /
  key contract the intake says to preserve **exactly**, and would break wf3's
  value-identity by re-pointing rule 3.02's output at a wf1 rule.
- **Accepted cost:** one duplicate native-resolution extraction per project
  (wf1's `wf1_raw/` and wf3's `<key>/` both extract the same raw grid when their
  windows coincide). This is acceptable for CST's rapid first-order basin assessments
  (AGENTS.md Background) and is recorded as a consequence (§7). The "one store feeds
  all workflows" north star is explicitly deferred to **P3-2b/future** (§10 OQ-3), so
  it is on record and not silently dropped.

**Regridding / subcatchment aggregation approach — grid co-registration is the
crux.** The raw extraction is at the catalog **native resolution** — but *what* that
resolution is, and whether the extraction itself reprojects, is **branch-specific**
(risk-3):

- **era5 branch** (`extract_historical_climate.py:146-173`): a bare `get_rasterdataset`
  with **no** `reproject_like` — native ERA5 grid, no lapse/pressure correction.
- **chirps / chirps_global branch** (`extract_historical_climate.py:85-144`):
  **already reprojects** ERA5 variables onto the chirps grid (`reproject_like`, :119,
  :123), applies a **lapse-rate `temp()` correction** against a MERIT-Hydro DEM
  (:132-141, `lapse_rate=-0.0065`), and writes an orography sidecar. So on a chirps
  basin the "raw" extraction is neither native-ERA5 nor correction-free.

The test fixture is **era5** (`config/workflows/snake_config_model_test.yml:24`), so the
fixture and the era5-branch reasoning below hold for it. gabon's `clim_source` is
**unconfirmed at design time**; if gabon is chirps/chirps_global the extraction's own
reproject+lapse changes the value story and the expected delta — §5.5 carries a
branch-aware acceptance note for that case.

The subcatchment raster (`staticmaps.nc["subcatchment"]`) is on the **model grid**
(`model_resolution` default `0.00833°`, `Snakefile_model_creation:35`) — far finer than
native era5/chirps. `climate_forcing_by_subcatchment` aggregates via
`ds.groupby(subcatchment.rename("index")).mean()` on the **shared 2-D (lat, lon) grid**
(`climate_forcing.py:52-55`). In the **old** path this works because `mod.forcing.data`
and `mod.staticmaps.data["subcatchment"]` are both on the model grid, co-registered by
construction. In the **new** path the raw climate grid and the model-grid zone raster do
**not** share coordinates, so a direct raster `groupby` of native-grid climate by a
model-grid zone array is ill-defined — xarray cannot align mismatched (lat, lon) coords.
**A direct native-grid raster groupby is mechanically non-viable** (Option A, §6.3). The
aggregation must therefore either (a) **reproject the raw climate to the model grid
first**, then reuse the existing raster `groupby` (Option B), or (b) compute **zonal
statistics by subcatchment polygon** (Option C). §6.3 re-evaluates both on the corrected
facts; **the selected mechanism is now Option B (reproject-first)** — the v1 selection of
Option C is reversed (see §6.3 and the risk-4/arch-2 rationale). Option B needs no
subcatchment-polygon vector (which does not exist as an artifact — risk-4) and keeps the
`climate_forcing_by_subcatchment(forcing, subcatchment)` DataArray contract intact.

**What the value change actually is — corrected (risk-1).** The v1 draft asserted the
old↔new delta is "regridding/method-scale" and that a delta "far exceeding the
native↔model resolution ratio" is a failure signal. **That characterization is wrong**,
because it mis-describes what the *old* path aggregates. The old-path forcing
(`mod.forcing.data` = `inmaps_historical.nc`) is **not** regridded raw ERA5. It is built
by `setup_time_horizon.py` via hydromt's `setup_precip_forcing` +
`setup_temp_pet_forcing` with `temp_correction=True`, `press_correction=True`,
`dem_forcing_fn=<oro_source>`, and `pet_method="debruin"` (era5) / `"makkink"` (eobs),
all computed **on the model grid** (`setup_time_horizon.py:56-70`). So old-path
subcatchment temperature is **lapse-rate + pressure corrected to the model DEM**, and
old-path PET is computed by a specific method on that corrected grid.

The new era5 path aggregates **raw native-grid ERA5 with no lapse/pressure correction**
(`extract_historical_climate.py:159-173`) and PET must be derived separately (OQ-2). So
the old↔new subcatchment delta conflates **three distinct effects**, not one:

1. **regridding** (native ERA5 → model grid), the only effect v1 named;
2. **temperature lapse/pressure correction** — a *systematic* term ≈
   `-0.0065 °C/m × (mean model-DEM elevation − ERA5-orography elevation)`; a few hundred
   metres of relief in a subcatchment gives a **1–3 °C offset** that has nothing to do
   with regridding scale;
3. **PET-method choice** (the EP panel), if the wf1-side PET method differs from the
   model's `debruin`/`makkink`.

**Selected resolution — reproduce the model's own regrid + correction + PET method on
the wf1 raw path (model parity), with the exact callables and a per-branch pipeline
pinned here (ext1-1, ext1-2).** Under parity the plots represent **what actually drives
the model** (the point of the re-source), and the residual old↔new delta is genuinely
regridding/aggregation-scale, so §5.5's gate can reason about it. Parity is implemented
by `model/climate_parity.py::model_parity_climate(ds_raw, dem_model, dem_forcing,
pet_method)` (§5.1), which reuses the **same public hydromt process functions the
forcing build itself calls** — verified against the installed packages (hydromt 1.3.1,
hydromt_wflow 1.0.2 in the pixi env) and cited by file:line so implementation cannot
drift onto invented APIs. The EP panel is **retained unconditionally** — the mechanism
below is confirmed feasible on plain gridded data, so no output change beyond the
sanctioned re-source occurs (the v2 "drop the EP panel if infeasible" escape hatch is
**deleted**; OQ-2 is RESOLVED).

**The exact callables (all plain-xarray, model-free — `hydromt.model.processes.meteo`,
the same module the forcing build delegates to at `hydromt_wflow/wflow_sbm.py:3667,
3707,3310`; our extraction already imports from it today,
`extract_historical_climate.py:14`):**

1. **Precip** — `hydromt.model.processes.meteo.precip(precip, da_like, clim=None,
   freq=None, reproj_method="nearest_index", resample_kwargs=None)`
   (`meteo.py:47`). Parity call mirrors `setup_precip_forcing`
   (`wflow_sbm.py:3310-3316`): `precip(ds_raw["precip"].astype("float32"),
   da_like=dem_model, clim=None, freq=pd.to_timedelta(86400, unit="s"),
   resample_kwargs=dict(label="right", closed="right"))`. Input `[mm]` daily; output
   `precip [mm]` on the model grid, clipped ≥ 0. `clim=None` matches our build
   (no `precip_clim_fn` in `config/templates`).
2. **Temp** — `hydromt.model.processes.meteo.temp(temp, dem_model, dem_forcing=None,
   lapse_correction=True, freq=None, reproj_method="nearest_index",
   lapse_rate=-0.0065, resample_kwargs=None)` (`meteo.py:115`). Parity call mirrors
   `setup_temp_pet_forcing` (`wflow_sbm.py:3667-3673`): `temp(ds_raw["temp"],
   dem_model=dem_model, dem_forcing=dem_forcing, lapse_correction=True, freq=None)`.
   Input `[°C]`; internally: subtract `lapse_rate × dem_forcing` (temp at quasi-MSL),
   `reproject_like(dem_model, "nearest_index")`, add `lapse_rate × dem_model`
   (`meteo.py:160-185`). After PET, temp is time-resampled exactly as the build does
   (`resample_time(..., freq=86400s, downsampling="mean", label="right",
   closed="right")`, `wflow_sbm.py:3732-3740`).
3. **PET** — `hydromt.model.processes.meteo.pet(ds, temp, dem_model,
   method="debruin", press_correction=False, wind_correction=True, wind_altitude=10,
   reproj_method="nearest_index", freq=None, resample_kwargs=None)` (`meteo.py:323`).
   Parity call mirrors `wflow_sbm.py:3706-3718`:
   `pet(ds_raw[["press_msl", "kin", "kout"]], temp=temp_out, dem_model=dem_model,
   method=pet_method, press_correction=True, wind_correction=True, wind_altitude=10,
   reproj_method="nearest_index", freq=pd.to_timedelta(86400, unit="s"),
   resample_kwargs=dict(label="right", closed="right"))`. Required inputs for
   `debruin`: `press_msl [hPa]`, `kin [W m-2]`, `kout [W m-2]` (for `makkink`:
   `press_msl`, `kin` only) — all already written by the extraction. `temp` must be
   on the model grid identical to `dem_model` (`meteo.py:389-390`) — satisfied, it is
   step 2's output. Internally: `ds` is reprojected to the model grid (`meteo.py:393`),
   pressure is lapse-corrected **once** via `meteo.press(...)` (`meteo.py:396-403`,
   `press_correction(dem_model)` multiplicative factor), then
   `pet_debruin(temp, press, kin, kout, timestep=86400)` (`meteo.py:419-426`).
   Output `pet [mm]`.

**Inputs to the parity module (loaded by `plot_results.py`, model-side per C1):**
`dem_model = staticmaps.nc["land_elevation"]` — the same staticmap the build passes as
`dem_model` (`self.staticmaps.data[self._MAPS["elevtn"]]`, `wflow_sbm.py:3669`;
`"elevtn" → "land_elevation"` per `hydromt_wflow/naming.py:10`). `dem_forcing` is
branch-specific (table below): on era5, the `era5_orography` catalog source
fetched exactly as the build does (`get_rasterdataset(oro_source, geom=ds.raster.box,
buffer=2, variables=["elevtn"]).squeeze()`, `wflow_sbm.py:3658-3665`) — rule 1.10
therefore gains `params: data_sources = DATA_SOURCES, clim_source` — or, on the chirps
branch, the extraction's orography sidecar received as the **declared `oro_nc`
rule-1.10 input** (ext2-1; eobs is excluded from this path, ext2-3). `pet_method` and `oro_source` derive
from `clim_source` by the **same mapping `prep_hydromt_update_forcing_config` uses**
(`setup_time_horizon.py:19-26`: eobs → (`eobs_orography`, `makkink`); everything else →
(`era5_orography`, `debruin`)). All variables are cast `float32` first, matching the
build (`wflow_sbm.py:3654-3655`). The build's final `where(mask)` (basins mask =
the `subcatchment` staticmap, `naming.py:43-46`) needs no parity step: the groupby
aggregation by `subcatchment` already restricts to exactly those cells.

**Per-`clim_source` transformation pipeline (ext1-1) — which corrections extraction
has already applied, which wf1-side transformations remain, and the exactly-once
ledger:**

| Branch | Extraction has already applied (shared function, unchanged) | wf1-side remaining (parity module) | Net-correction ledger (must hold) |
| --- | --- | --- | --- |
| **era5** (test fixture, `snake_config_model_test.yml:24`) | Nothing — bare native-grid `get_rasterdataset`, no reprojection, no corrections (`extract_historical_climate.py:159-173`). | Calls 1-3 above with `dem_forcing` = `era5_orography` (catalog), `pet_method="debruin"`. | Temp lapse: 0 (extraction) + 1 (parity) = **1**, referenced era5-orography → model DEM — identical to the build's own path. Pressure: **1** (inside `meteo.pet`). Precip: **0**. |
| **chirps / chirps_global** | Precip: chirps native, untransformed. Temp/temp_min/temp_max: era5, **already lapse-corrected onto the chirps grid** against the MERIT DEM via the same `meteo.temp(..., dem_forcing=era5_orography, lapse_correction=True)` (`extract_historical_climate.py:132-141`). press_msl/kin/kout: era5 `reproject_like` → chirps grid, `nearest_index` — resampled, **never physically corrected**. Writes the DEM it corrected to as the sidecar `{clim_source}_orography.nc` (`:143-144`). | Calls 1-3, **but `dem_forcing` MUST be the extraction's orography sidecar** (the MERIT-on-chirps-grid DEM), NOT `era5_orography`. `meteo.temp` then first subtracts `lapse_rate × sidecar_dem` — **exactly inverting** the correction the extraction embedded — reprojects, and re-adds `lapse_rate × dem_model`. `pet_method="debruin"` (chirps configs force temp/PET from era5, `setup_time_horizon.py:23-26`). The sidecar is a **declared output of the extraction rule** (`oro_nc = wf1_raw/orography.nc`, stable path — the wrapper relocates the shared function's `{clim_source}_orography.nc` to it) and a **declared branch-resolved input of rule 1.10**, never discovered as a sibling of `climate_nc` (ext2-1). | Temp lapse net: +1 (extraction, → chirps DEM) − 1 (parity MSL-shift with the *same* DEM) + 1 (parity, → model DEM) = **1**. Passing `era5_orography` here instead would leave the chirps-DEM correction unreversed ⇒ **double correction** — the ext1-1 defect; the sidecar is therefore *mandatory*, asserted by the unit test below. Pressure: **1**. Precip: **0**. |
| **eobs** | **Excluded from P3-2a (ext2-3, arbitrated — exclude variant).** eobs is not a supported source for the wf1 raw path: `Snakefile_model_creation` raises an explicit configuration error at **DAG-parse time** when the config sets `clim_historical: eobs` — message naming the exclusion and the supported set (`"clim_historical: eobs is not supported by the P3-2a wf1 raw-climate path; supported sources: era5, chirps, chirps_global"`) — so the failure is early and loud (before any rule executes, and it reds every dry-run), not a mid-run extraction error. The wf1 wrapper asserts the same as a belt-and-braces guard. | n/a — **no eobs parity pipeline is specified in P3-2a** (per the arbitration ruling, deliberately not designed here). | n/a. Restoring eobs is a separate future scoping item (verify the catalog carries the extraction variables, then design its parity row); until then the bounded support statement below governs. |

**Bounded support statement (ext2-3).** The wf1 raw-climate path (the new
extraction rule + `model/climate_parity.py` + the re-sourced rule-1.10 plots)
supports exactly **era5, chirps, and chirps_global**. `clim_historical: eobs` is
rejected with the parse-time configuration error above; no eobs pipeline is
specified or implied by this design. wf3's use of the shared extraction module is
untouched by this exclusion (the guard lives in `Snakefile_model_creation` and the
wf1 wrapper only), so wf2/wf3 value-identity (C2/C3) is unaffected. References
elsewhere in this document to the `setup_time_horizon.py:19-26`
eobs → (`eobs_orography`, `makkink`) mapping describe the *existing forcing build*,
not a P3-2a-supported wf1 raw-path branch.

**Exactly-once invariant and its tests (ext1-1).** The invariant: *per branch and
variable, the net count of physical corrections between raw source data and the
plotted grid is exactly the ledger above — temp lapse net 1 (referenced to the model
DEM), pressure 1, precip 0.* Proven two ways:

- **Unit test (synthetic, fast — commit 3):** on a small synthetic grid with known
  DEMs, (a) era5-branch: assert parity-output temp equals
  `T_raw + lapse_rate × (dem_model − dem_forcing)` within float tolerance (away from
  edges); (b) chirps-branch: apply the extraction-style correction to a synthetic
  "chirps DEM" first, then the parity step with that DEM as sidecar, and assert the
  result equals the *single-step* correction to the model DEM within tolerance — a
  stacked (double) correction fails this by `lapse_rate × dem_chirps` ≈ °C-scale.
- **Integration invariant (fixture, era5 — commit 5 QA, the grid-level check G of
  §5.5):** the parity grid `{precip, temp, pet}` is compared against
  `inmaps_historical.nc` (still built by rule 1.08) within the basin mask: per-variable
  max-abs ≤ tolerance (temp ≤ 0.05 °C, precip/pet ≤ 0.05 mm d⁻¹; float32 +
  domain-clip headroom — actuals recorded in the baseline note). The parity path and
  the build execute the *same* functions on the *same* sources, so a double or missing
  correction shows as `lapse_rate × Δdem` ≈ 1-3 °C ≫ tolerance and reds the gate.

PET is derived **wf1-side only** (never in the shared `prep_historical_climate`, which
wf3 rule 3.02 also calls — adding `pet` there would change wf3's
`extract_historical.nc` and break C2/C3). The extraction writes `precip`, `temp`,
`temp_min`, `temp_max`, `kin`, `kout`, `press_msl` but **not** `pet`
(`extract_historical_climate.py:164-173`), while `climate_forcing_by_subcatchment` /
`plot_clim` require `precip`, `temp`, **`pet`** (`FORCING_TO_CLIM`,
`climate_forcing.py:10-14`) — the parity module's step 3 closes exactly that gap with
the build's own method.

### 5.3 Mechanical rewire of wf2/wf3 imports (value-identical)

The lift is three file moves; each moved file keeps its `__main__` /
`snakemake.*`-reading block **verbatim** (only the internal import prefix of its own
package changes if it imports a sibling). The rewire surface:

- **wf3 rule 3.02 `extract_climate_grid`** (`Snakefile_climate_experiment:194-215`):
  `script:` path flips from `blueearth_cst/experiment/extract_historical_climate.py`
  → `blueearth_cst/climate_analysis/extract_historical_climate.py`. This is a
  **rewire site, not a value change** — Snakemake resolves `script:` relative to
  `workflow.basedir`, so it is a pure string substitution. All `input:`/`output:`/
  `params:` (incl. `guard_ok = ancient({store_dir}/.guard_ok)`, the keyed
  `climate_nc` output, the region input) stay **byte-identical**. The P3-1 keying and
  `.guard_ok` wiring is untouched (C3).
- **wf3 catalog prep — rule 3.08 `climate_data_catalog`** (the sole consumer of
  `prepare_climate_data_catalog.py`, `Snakefile_climate_experiment:315-337`, `script:`
  at :337): `script:` path flips `blueearth_cst/projections/...` →
  `blueearth_cst/climate_analysis/...`. The `oro_path` param (pointing under the keyed
  store dir, `:331`) and all other params stay verbatim. **wf2
  (`Snakefile_climate_projections`) does not consume the catalog module at all** — the
  sole consumer is wf3 rule 3.08 (arch-5).
- **The new wf1 extraction rule does NOT execute the moved module's `__main__` block —
  it targets a dedicated wf1 wrapper script (ext1-4, SELECTED).** The moved module's
  `__main__` reads `sm.input.prj_region`, `sm.output.climate_nc`, and
  `sm.params.{data_sources,clim_source,starttime,endtime}`
  (`extract_historical_climate.py:193-198`); the wf1 rule declares `basin_nc =
  ancient(staticmaps.nc)` instead of `prj_region` (arch-1), so pointing the wf1 rule's
  `script:` at the moved module would `AttributeError` on `sm.input.prj_region` at
  execution — and patching conditional Snakemake-shape dispatch into the supposedly
  verbatim moved module would risk wf3 drift. **The contract, pinned:**
  - **File:** `blueearth_cst/model/extract_climate_wf1.py` (NEW, §5.1) — the wf1
    rule's `script:` target. It imports
    `from blueearth_cst.climate_analysis.extract_historical_climate import
    prep_historical_climate`, reads `sm.input.basin_nc`, `sm.output.climate_nc`,
    `sm.params.{data_sources,clim_source,starttime,endtime}`, derives the bbox from
    `staticmaps.nc` (§5.2), and calls `prep_historical_climate(region_fn=None,
    fn_out=..., data_libs=..., clim_source=..., starttime=..., endtime=...,
    bbox=bbox)` under the standard `tee_to_log` pattern. On the
    chirps/chirps_global branch it additionally reads `sm.output.oro_nc` and, after
    the call, moves the shared function's `{clim_source}_orography.nc` sidecar to
    that declared path (ext2-1, §5.2). It asserts `clim_source` is not `eobs` as
    the belt-and-braces guard behind the parse-time configuration error (ext2-3,
    §5.2).
  - **The moved module keeps its wf3 `__main__` block verbatim**; its `script:` path
    is referenced only by wf3 rule 3.02. No conditional dispatch is added to it.
  - **Reconciled surfaces:** the §5.2 rule declaration (`script:` names the wrapper),
    the §5.1 layout (wrapper listed under `model/`), §8 commit 3 (wrapper + its unit
    tests land there), and the tests (`test_cli.py` dry-run resolves the wrapper path;
    the bbox unit test targets the wrapper's derivation function; the commit-6 e2e
    executes it).
- **Any Python import** of the moved modules
  (`from blueearth_cst.experiment.extract_historical_climate import ...`,
  `from blueearth_cst.model.climate_forcing import ...`,
  `from blueearth_cst.projections.prepare_climate_data_catalog import ...`) is
  rewritten to the `climate_analysis` prefix. Known live site: `plot_results.py:26`
  imports `climate_forcing_by_subcatchment` — rewired to
  `from blueearth_cst.climate_analysis.subcatchment_climate import
  climate_forcing_by_subcatchment` (function name kept, OQ-1). A grep-derived inventory
  (§8 commit 1) enumerates every import + test import site.
- **Tests** that import the moved modules move/rewire per the same table.
  `test_extract_historical_climate.py` and `test_prepare_climate_data_catalog.py`
  keep their assertions verbatim (value-identity — those functions are unchanged).
  **`tests/test_climate_forcing.py` is a second live caller of
  `climate_forcing_by_subcatchment`** (lines 48/53/61/70, DataArray fixtures with
  `_FillValue` nodata, repo-1). Under the selected **Option B** the function's signature
  and raster-`groupby` algorithm are **unchanged**, so this test's assertions **do stay
  verbatim** — it is only *repointed* to the new import path in the move commit (commit
  1). (This corrects the v1 draft, which had selected Option C — a GeoDataFrame /
  polygon-zonal reduction that would have *required rewriting* this test; Option B avoids
  that rewrite entirely. See §6.3.)

**Value-identity check.** After the rewire, the semantic tree diff
(`dev/scripts/semantic_tree_diff.py`) over a wf2+wf3 run must be **clean modulo
nothing** (wf2/wf3 produce no changed value — only `script:` string paths and Python
import lines change, neither of which appears in output artifacts). Only the wf1
subcatchment climate plots change (§5.5).

### 5.4 Backward-compat / deprecation posture for old module paths

**Selected: transitional re-export shims at the old module paths, removed in a final
cleanup commit within this milestone.**

Because imports are `from blueearth_cst.<stage>.<module> import ...`, a bare `git mv`
breaks every importer and every `script:` path the instant it lands (decision-record
migration rule). Two ways to keep per-commit runnability (C5):

- **Option A — atomic move + reference-rewrite in one commit.** Move all three files
  and rewrite every import + `script:` path in the same commit, justified as one
  mechanical transform.
- **Option B — transitional re-export shim (SELECTED).** Commit 1 creates
  `climate_analysis/` with the real code and leaves the old paths as thin shims
  (`from blueearth_cst.climate_analysis.<module> import *  # moved — P3-2a`).
  Subsequent commits repoint `script:` paths and Python imports to the new home. A
  final commit deletes the shims once no live citer remains.

**Why B.** The move touches Snakemake `script:` paths (runtime-only, `--dry-run`-blind
for `script:` bodies in the sense that the DAG resolves the path but does not execute
the module) *and* Python imports across modules and tests. A shim window lets each
rewire land as its own small reviewable commit with the tree runnable throughout,
and any missed citer resolves one hop away rather than red-ing the build — the same
transitional-safety-net logic the ADR consolidation guidance uses. The shims are
**this milestone's** artifact and are **deleted before the milestone closes** (they
are not a permanent compat layer — no external consumer imports these internal paths;
the platform surface is the Snakefiles, unchanged). The old paths are **not** a §7
contract surface (they are internal package modules, not `rule all` filenames or
user-facing config keys), so their removal needs no migration note beyond the
`MIGRATION.md`-style old→new map recorded in the milestone record.

*(Naming note: "Option A/B" here are **deprecation-posture** options, unrelated to the
aggregation-mechanism Option A/B/C in §6.3.)*

**Shim correctness depends on the moved modules carrying no `__all__` (repo-4).** The
star-import shim (`from M import *`) keeps the monkeypatch tests green through the shim
window only because the moved modules declare no `__all__`:
`test_extract_historical_climate.py:338` and `test_prepare_climate_data_catalog.py:66,81`
patch `ehc.hydromt.DataCatalog` / `pcdc.hydromt.DataCatalog` — an attribute on the shared
`hydromt` module object. With no `__all__`, `from M import *` re-binds that shared module
into the shim namespace, so the patch bites through. Mitigation: **repoint the moved-module
tests to the new import path in the same move commit (commit 1)** so the shim is never
load-bearing for the test suite; and note the no-`__all__` dependency so a future edit does
not silently break it.

**`script:`-path repoints are dry-run-blind; add an execution check before shim deletion
(risk-6).** `--dry-run` resolves the `script:` path (so a *missing* path reds the DAG) but
does **not execute** the script body; `test_cli.py` dry-runs, it does not run the scripts.
So a *missed* `script:` repoint could survive the per-commit dry-run gate and only surface
at execution after shim deletion (commit 6). Mitigation: the commit-6 gate adds an
**execution-level check** — the e2e `run_workflows.py` run (§9) covering the two `script:`
sites (wf3 rule 3.02, rule 3.08) plus the new wf1 rule — before/at shim deletion, rather
than relying on dry-run alone for the `script:` repoints.

**Rejected — leave permanent shims.** Rejected: dead re-export modules accumulate
and the "model-independent subpackage" boundary is muddied by ghost paths under
`experiment/` and `projections/`. Removal is cheap once citers are repointed.

### 5.5 Acceptance-gate implementation (side-by-side QA + characterized diff)

Per intake decision 4, the plot value change is accepted via **visual QA +
characterized diff**, both computed on the test fixture (and gabon when available).

**Side-by-side old/new plots.** Produce the `clim_{station}_{period}.png` plots
(`clim_wflow_1_month.png`, etc. — `func_plot_signature.py:772`) **both ways** on the
test fixture:

- *old*: on the pre-P3-2a tip (or a pinned copy of the ADR-0002 path), the plots
  built from `mod.forcing.data` (lapse/pressure-corrected, PET-derived model forcing on
  the model grid) aggregated per subcatchment.
- *new*: on the P3-2a tip, the plots built from the raw extraction, reprojected to the
  model grid with the model's own correction + PET method (§6.3 Option B), aggregated
  per subcatchment.

Both sets are laid side by side for user sign-off at the milestone gate. The plots
are QA artifacts, **not** manifested (§ manifest scope below), so this is a visual
comparison, not a byte match.

**Characterized diff — component-decomposed, not "regridding-scale" (risk-1, arch-2,
arch-3).** The v1 draft attributed the whole old↔new delta to "regridding/zonal-method"
and set a "delta far exceeding the resolution ratio ⇒ failure" heuristic. That is
**dropped**: as §5.2 establishes, the old path is corrected, PET-derived model forcing,
so the raw delta conflates **three** effects — regridding, temp lapse/pressure
correction, and PET-method choice. The corrected acceptance logic:

- **Operational comparison ladder (ext1-3).** The decomposition is defined as deltas
  between **named, persisted states**, not as a post-hoc attribution. Let `R(·)` be the
  common reduction `climate_forcing_by_subcatchment(·, subcatchment)` followed by a
  per-subcatchment **monthly climatology** (12 months × `index` × {P, T, EP}). States,
  all produced by a dev-side compare script (`dev/milestones/p32a/compare_climate_ladder.py`,
  commit 5) except where noted:
  - **S0** — `inmaps_historical.nc` (production artifact, rule 1.08): the old path's
    grid. **A0 = R(S0)** is exactly the old plot input.
  - **S1** — `climate_historical/wf1_raw/extract_historical.nc` (production artifact,
    the new extraction rule): the raw branch-native extraction. S1 is **not** a ladder
    rung reduced by `R` — native-grid data cannot be grouped by the model-grid zone
    raster (the very reason Option A was rejected, §6.3). Its branch-specific embedded
    preprocessing (chirps: lapse-corrected temp; era5: none) is a documented property
    per the §5.2 branch table, covered by the exactly-once unit test — it does not add
    rungs.
  - **S2** — *regridded-only*: S1 put on the model grid by the §5.2 callables with
    corrections **off** (`meteo.temp(..., lapse_correction=False)`, `meteo.pet(...,
    press_correction=False)` — which renames `press_msl → press` uncorrected,
    `meteo.py:404-406` — same `pet_method`, same freq/resample kwargs). Persisted:
    `qa/l1_regrid_only.nc`. **A1 = R(S2)**.
  - **S3** — *regrid + corrections + PET* = the production parity grid (§5.2, all
    corrections on). Persisted: `qa/l2_parity.nc`. **A2 = R(S3)** is exactly the new
    plot input.
  Components, each the delta between **adjacent states** (per variable × subcatchment
  × month; summarized as mean, max-abs, RMSE — max-abs and RMSE so a localized offset
  is not averaged away):
  - **Correction component = A2 − A1** (temp lapse + pressure-through-PET; the
    correction×regrid interaction is included by construction, since regrid is on in
    both states and the correction order is the build's own fixed order, §5.2 — no
    ambiguous interaction term remains).
  - **Precip null-check:** `A2 − A1 ≡ 0` for P, asserted **exactly** (no correction
    touches precip on any branch) — a nonzero value is a pipeline bug, not a finding.
  - **End-to-end sanctioned change = A2 − A0** — the number the user signs off.
  - **Grid-level parity check G** = per-variable max-abs/RMSE of `S3 − S0` on the model
    grid within the basin mask (the §5.2 integration invariant). G localizes any
    residual *before* aggregation can smear it.
  - **Drill-down rungs (on demand only):** if G or `A2 − A0` exceeds tolerance and the
    cause is not evident, two extra states isolate the correction pieces — S3a (temp
    correction on, pressure off) and S3b (pressure on, temp off). They are diagnostic,
    not part of the standard record.
  **Attribution rule for the validator:** every reported residual must be assigned to a
  named ladder edge (`A2 − A1`, `A2 − A0`, or G) — never to a free-floating mechanism.
- **Expected magnitudes, parity-grounded and branch-aware (risk-1, risk-3):**
  - **era5 (the test fixture, `snake_config_model_test.yml:24`):** the parity path and
    the forcing build execute the same functions on the same sources, so
    `A2 − A0 ≈ 0` and G ≤ the §5.2 tolerances (float32/domain-clip headroom). A large
    P/T/EP delta *is* a failure signal (units mismatch, wrong DEM, mis-set correction
    flags, wrong PET method) — G plus the ladder says which. `A2 − A1` shows the
    expected physical terms: T ≈ `-0.0065 °C/m × (model DEM − era5 orography)` per
    subcatchment, EP the corresponding PET response, P ≡ 0.
  - **chirps/chirps_global:** the extraction already reprojects era5 onto the chirps
    grid and lapse-corrects temp (`extract_historical_climate.py:106-144`), so the new
    path's temp travels era5 → chirps grid → model grid while the build's travels
    era5 → model grid directly. With the sidecar-DEM chained correction (§5.2 table)
    the correction algebra cancels exactly; the remaining `A2 − A0` / G residual is an
    **interpolation-hop residual** (one extra `nearest_index` hop for temp/press/kin/
    kout). **"Small" is not left subjective (ext2-2, arbitrated — tolerance variant;
    the direct-hop reference-state rung was explicitly not chosen and is not
    added):** chirps acceptance is governed by **pinned, variable-specific
    tolerances established from a pinned fixture run**, via this defined
    defer-and-pin procedure:
    1. **Pinning (first chirps basin).** P3-2a's test fixture is era5 and the repo
       carries no chirps fixture, so chirps tolerances cannot be honestly pinned in
       this milestone; they are pinned on the **first chirps/chirps_global basin**
       run through the commit-5 ladder. That run is the pinned fixture: its config
       and inputs are recorded (G4 recorded-state discipline), the ladder is
       computed, and the per-variable (P, T, EP) max-abs and RMSE on the
       `A2 − A1`, `A2 − A0`, and G edges are recorded in the `dev/milestones/p32a/` baseline
       note as the **chirps tolerance set**.
    2. **Justification requirement.** Each pinned tolerance must be justified
       against the hop mechanism, not merely observed: the recorded value is
       checked for order-of-magnitude consistency with a stated bound — one
       `nearest_index` cell displacement (≤ one chirps source cell, ≥ 0.05°) times
       the local per-cell field gradient for that variable — and is **exactly 0
       for P** (chirps precip takes no hop: it is read natively in both extraction
       and parity). A candidate tolerance that cannot be reconciled with that
       bound is itself a gate failure to investigate, not a value to pin.
    3. **Gate.** Until the tolerance set is pinned, **no chirps-basin re-sourced
       plots may be accepted** — the acceptance gate for a chirps basin is
       *blocked*, never waved through on "expected small hop residual" language.
       From the pinning run onward, any chirps-basin residual on those edges
       **exceeding the pinned tolerances without an assigned, named cause fails
       the gate** — a wrong DEM, source-field mismatch, or correction error
       surfaces exactly as such unexplained excess (the ext2-2 failure mode).
    gabon's `clim_source` must be confirmed before its diff is interpreted; if
    chirps, gabon is the natural pinning candidate and runs the procedure above
    rather than a free-form "expect a hop residual" reading.
- The characterized diff (per-variable, per-component delta table + summary) is recorded
  in the **milestone's baseline record** (`dev/milestones/p32a/` baseline note, R3/R5
  `baseline_diffs.md` style). **No a-priori threshold gate** (intake decision 4 rejected
  an arbitrary number); the gate is user sign-off on the visual + the recorded
  component-decomposed characterization.

**Manifest re-record scope — no-op, and a knowing divergence from intake decision 4
(repo-2, repo-3).** Intake decision 4 says the acceptance flow ends with "the manifest
then re-records the changed targets." Verified: the wf1 baseline manifest
(`dev/scripts/check_baseline.py` `TARGETS`, model_creation entries = `hydro_wflow_1.png`,
`basin_area.png`, `precip.png`, `snake_config_model_creation.yml`, `output.csv`)
fingerprints **only** those; the changed `clim_wflow_*_*.png` plots are **NOT** in
`rule all` (`Snakefile_model_creation:54-61`) and **NOT** manifested. (`outlet_index.csv`
is a `rule all` target at `:60` but is **not** a `check_baseline` TARGET — corrected from
the v1 draft, which listed it in the manifested set, repo-3.) So no re-record of the
changed targets is *possible*: the one sanctioned value change produces **no manifested
artifact**, and the characterized diff — not the manifest — carries the full acceptance
weight. This is a **justified divergence from the anchor's literal wording** (the
mechanism does not apply because the target is unmanifested), which the human gate must
**knowingly accept** rather than read the anchor as satisfied; OQ-4 tracks the future
manifest-extension path if the `clim_*` plots are ever promoted to `rule all`. Therefore:

- The changed outputs (the `clim_*` plots) are **outside the manifest slice**, so the
  manifest **does not need a re-record for the value change** — mirroring ADR 0002's
  own "`output.csv` untouched → manifest unaffected" logic.
- The **new wf1 extraction output** (`climate_historical/wf1_raw/extract_historical.nc`)
  is a new intermediate netCDF, not in `rule all` — also unmanifested.
- **The manifested wf1 targets must stay byte/tolerance-identical** (C2): the
  hydrograph, map, and forcing-map PNGs and `output.csv` are untouched by this change
  (the change is confined to §4 of `plot_results.py` and a new leaf rule). The gate
  **still runs `check_baseline check`** on the manifested slice to prove those did
  not drift — the manifest re-record is a **no-op confirmation**, and the
  characterized diff carries the acceptance weight for the actual change.

If a later decision adds the `clim_*` plots to `rule all` (out of scope here), *then*
a manifest extension would be needed; flagged as OQ-4.

## 6. Alternatives considered

### 6.1 Subpackage boundary — move the plotting primitives too, vs leave them in `shared/`

- **Option — move `func_plot_signature.py` / `plot_map.py` into `climate_analysis/`**
  so all climate-analysis code sits in one package. **Rejected:** `plot_clim` and
  `plot_map` are already in `shared/` and are consumed by multiple stages (`plot_clim`
  by wf1 `plot_results`; `plot_map` by wf1 `plot_map`); `func_plot_signature` also
  carries `plot_hydro`/`plot_signatures`/`compute_metrics` which are **hydrograph/
  metrics** primitives, not climate. Moving the file would drag non-climate code into
  `climate_analysis/` or force a file split — churn with no decoupling win. R6 already
  scoped these as "genuinely zero-move" (§8). **Selected: leave in `shared/`.**
  Preferred-if: a future milestone splits `func_plot_signature.py` into climate vs
  hydrograph primitives — then the climate half could move. Out of scope here.

### 6.2 wf1 raw-climate source — reuse the P3-1 keyed store vs a separate wf1 extraction

- **Option — reuse `climate_historical/<key>/extract_historical.nc`** (the wf3 store)
  from wf1, since wf1 and wf3 share the `clim_source + historical_window` key.
  **Rejected on C3 + C4** (detailed in §5.2): pipeline order puts the store's producer
  (wf3, guard-gated) downstream of wf1, so a wf1 `input:` raises
  `MissingInputException` on a fresh project and inverts the documented order; making
  wf1 the producer re-architects the P3-1 `.guard_ok`/keying contract the intake
  preserves exactly. Preferred-if: a future milestone (P3-2b) promotes the keyed store
  to a **project-level, workflow-order-agnostic** climate store with its own guard —
  explicitly out of P3-2a scope.
- **Option — separate wf1 extraction to a wf1-owned path (SELECTED).** wf1 reuses the
  lifted *function*, not the store, writing to `climate_historical/wf1_raw/`. Cost:
  one duplicate native-resolution extraction per project — acceptable for rapid
  first-order assessments, recorded as a consequence.

### 6.3 Subcatchment aggregation — native-grid raster groupby vs reproject-first vs zonal-by-polygon

The `climate_forcing_by_subcatchment` reduction is a **raster `groupby`** of the climate
grid by the 2-D subcatchment-id array, requiring both to share (lat, lon) coordinates
(`climate_forcing.py:52-55`). Three mechanisms; **this section reverses the v1 ranking**
(v1 selected Option C; v2 selects **Option B**), on the corrected facts from risk-1,
risk-4, arch-2, and repo-1.

- **Option A — direct native-grid raster groupby.** Keep the function as-is and feed it
  the native-grid climate + the model-grid zone raster. **Rejected: mechanically
  non-viable** — native climate and model-grid zones do not share coordinates, so the
  `groupby` cannot align (§5.2). Only viable if the two grids were co-registered.
- **Option B — reproject the raw climate to the model grid first, then raster groupby
  (SELECTED).** Our own `reproject_like` of the raw climate onto the model-grid
  subcatchment raster, then the **existing, unchanged** `climate_forcing_by_subcatchment`.
  C1-clean (the reproject is *our* step over plain rasters, not the model's forcing-build
  artifact). **Why it is now selected over C:**
  1. **No non-existent input.** Option B needs only the raw climate raster + the
     model-grid subcatchment **raster** (`staticmaps.nc["subcatchment"]`, which exists).
     Option C needs a per-subcatchment **polygon GeoDataFrame** that **does not exist**
     as an artifact — `staticgeoms/` carries `basins/outlets/rivers/region/meta_basins`,
     **no** per-subcatchment polygon set (risk-4, arch-2) — so C silently requires an
     unspecified raster→vector polygonization that must preserve the integer-id →
     station-index mapping `plot_clim` consumes; a permuted mapping attaches plots to the
     wrong stations and the per-subcatchment diff would not catch it.
  2. **No open mechanism.** Option C's hydromt zonal-stats entry point was OQ-7-open
     (feasibility unconfirmed); Option B uses `reproject_like` (already used elsewhere in
     the same module, `extract_historical_climate.py:119,123`) + the existing groupby —
     no unproven API.
  3. **The delta is cleanly attributable, which is what the gate wants.** v1 rejected B
     for "deltas ~0, weaker decoupling story" — but risk-1 shows a delta attributable to
     a *known cause* (regridding + the model's own correction/PET, reproducible on the B
     path) is exactly what §5.5's gate needs. Reproducing the model's regrid on our own
     path (a) yields a residual delta that genuinely *is* regridding/aggregation-scale,
     (b) makes the plots represent **what drives the model**, and (c) keeps the
     `climate_forcing_by_subcatchment` signature + algorithm + **`test_climate_forcing.py`
     assertions** byte-stable (repo-1) — no test rewrite.
  The "weaker decoupling" objection is answered by §5.2: the decoupling is sourcing raw
  climate **independent of the model's stored `inmaps` forcing artifact**, not "avoid
  regridding." B does its own regrid over raw catalog data; it never reads the model
  forcing.
- **Option C — zonal statistics by subcatchment polygon (rejected here; deferred).**
  A vector zonal mean over subcatchment polygons sidesteps grid co-registration and is
  arguably the *most* model-independent end state, but it rests on the non-existent
  polygon input and the OQ-7-open API above, and it would force a `climate_by_zone`
  signature change (DataArray→GeoDataFrame) plus a full `test_climate_forcing.py` rewrite.
  **Deferred** to a future milestone that first provides a validated per-subcatchment
  polygon artifact with a guaranteed id→index mapping (a natural P3-2b/standalone-entry
  item). Recorded so the more-decoupled path is on the roadmap, not silently dropped.
- **Sub-option — cell-area weighting.** `climate_forcing.py:44-47` notes the aggregation
  is an **unweighted** cell mean, fine for small equatorial basins, less accurate at
  higher latitudes / large basins. P3-2a keeps unweighted (unchanged from today).
  Area-weighting is OQ-5, deferred.

### 6.4 Deprecation posture — hard atomic cutover vs transitional shim window

- **Option — hard atomic cutover** (one commit moves all files + rewrites all
  references). **Considered, not selected:** valid per the migration rule and simplest
  end state, but it makes commit 1 a large multi-file diff spanning three moved
  modules, ~all their importers, three-plus `script:` paths, and the tests — harder to
  review and to bisect if a rewire site is missed. **Selected: transitional re-export
  shims** (§5.4) so each rewire lands as a small reviewable commit with the tree
  runnable throughout, shims deleted before the milestone closes. Preferred-if the
  reviewer prefers a single atomic transform commit over a shim window — a legitimate
  taste call; flagged as OQ-6.

## 7. Consequences and risks

**Positive.**

- **The extraction (`prep_historical_climate`) becomes runnable without a built Wflow
  model** — region/catalog/window in, netCDF out, no model dependency. This is the piece
  the standalone climate-screening entry point and P3-2b can build on.
- The wf1 plots source the *actual raw climate* over the basin/window, decoupled from the
  model's stored `inmaps` forcing artifact.
- A settled `climate_analysis/` home exists for the deferred standalone entry point and
  for P3-2b to pin contracts around.
- Model-independence is **checkable** (C1 grep), not aspirational.

**Scope limit on "model-independent" (risk-2 — honest framing).** The **aggregation**
(`climate_forcing_by_subcatchment`) is model-independent *only in the C1 sense* ("takes no
`WflowSbmModel`"): it still needs the **subcatchment-zone raster**, which today exists only
as `staticmaps.nc["subcatchment"]` — a **built-model artifact**, loaded by the wf1-side
caller (§5.1/§5.2). So the aggregation half of the wf1 climate plots **cannot yet run
without a built model** for its zones. The headline "model-independent, checkable not
aspirational" claim is therefore true for **extraction**, not for **aggregation**. Closing
that gap (a model-independent per-subcatchment-zone source) is **out of P3-2a scope** and a
candidate for P3-2b / the standalone entry point; recorded so a reader — or P3-2b pinning
contracts around this layout — does not assume aggregation runs region+catalog+window-only.

**Negative / new obligations.**

- **One duplicate extraction per project** (wf1 `wf1_raw/` + wf3 `<key>/`). Extra runtime +
  disk on the wf1 build; acceptable for rapid assessments, removable in a future
  project-level-store milestone (OQ-3).
- **The wf1 subcatchment climate plots change value** — the sanctioned change, gated by
  §5.5. The delta is **not** a single "regridding scale": it decomposes into regridding +
  the model's temp/pressure correction + PET-method choice (§5.2/§5.5). Any downstream user
  comparing new plots to old must know this.
- **PET is a new wf1-side computation (OQ-2 — RESOLVED, ext1-2)** — the raw extraction
  does not write `pet`; the parity module derives it **wf1-side only** (never in the
  shared extraction function, which wf3 also calls) via the named callable
  `hydromt.model.processes.meteo.pet(...)` with the **same method the forcing build
  uses** (`debruin`/era5, `makkink`/eobs — §5.2), so the EP delta is attributable to
  regridding/aggregation, not method choice (arch-3). The EP panel is **retained
  unconditionally** — no unauthorized output change.
- **Two new `model/`-side modules** (`extract_climate_wf1.py`,
  `climate_parity.py`, §5.1) — small, wf1-owned, plain-xarray; rule 1.10 gains
  `data_sources`/`clim_source` params so the parity module can fetch the orography
  source from the catalog on the era5 branch (on chirps the DEM arrives as the
  declared `oro_nc` input instead — ext2-1, §5.2).
- **Subcatchment-zone raster needed wf1-side** — the selected raster path (§6.3 Option B)
  needs the model-grid subcatchment **raster** (`staticmaps.nc["subcatchment"]`), read by
  the wf1 caller and passed as an xarray DataArray. (No per-subcatchment polygon vector is
  needed under Option B — that was the rejected Option C's requirement.)
- **Transitional shims** exist mid-milestone (removed before close).

**Neutral (must be planned for).**

- New wf1 rule renumbers subsequent `1.NN` comment headers + log/benchmark prefixes
  (mechanical sweep; ephemeral untracked paths — not a contract surface, per naming.md §9).
- `plot_results` rule 1.10 loses its `forcing_path` input and gains the raw-climate input —
  a DAG edge change (honest dependency), `--dry-run`-visible. The `script = ...` input label
  (`:217`) is left intact (repo-5).
- Optional module rename `climate_forcing.py` → `subcatchment_climate.py` (OQ-1) touches the
  one importer (`plot_results.py:26`) and the test-import line; the function name is kept
  (§5.1), so no assertion changes.

**Risks.**

- **R1 — value change not attributable to a known cause.** If, on the selected Option B
  with model-parity, the residual delta is *not* regridding/aggregation-scale (a units
  mismatch, a wrong DEM, mis-set correction flags, or a mismatched PET method), the
  component-decomposed characterized diff (§5.5) catches it — that is the gate's job.
  Mitigation: §5.5 reports per-component max-abs and RMSE, not a single lumped number;
  user sign-off on the visual.
- **R2 — accidental wf2/wf3 value drift.** A missed import/`script:` rewire or a stray
  edit to a moved module's body. Mitigation: semantic-tree-diff clean requirement (C2)
  + `check_baseline` on the manifested wf2/wf3 slice (noting the manifest is stale/
  mixed-provenance per `baseline-manifest-coverage` memory — pre-existing wf2/wf3 drift
  unrelated to P3-2a may show; the gate distinguishes it via the pre/post run-relative
  compare).
- **R3 — P3-1 contract nick.** Any change to rule 3.02's `input:`/`output:`/`params:`
  or the `.guard_ok`/keyed-store paths. Mitigation: §5.3 pins the rewire to the
  `script:` path only; a diff of rule 3.02 pre/post must show *only* the `script:`
  string changed.

## 8. Migration + commit plan

**Branch.** One task branch off the P3-2a milestone line per the repo branch model
(`branch-model` memory: one standing branch per milestone; `task/p32a-*` for the work,
merged to the milestone branch/`main`). Commit prefix `p32a:` (registered in the
roadmap prefix list, `dev/roadmap.md`).

**Commit sequence (small, reviewable, each `--dry-run`-clean + `test_cli.py`-green):**

1. `p32a: create climate_analysis subpackage + shim old module paths` — add
   `blueearth_cst/climate_analysis/{__init__,extract_historical_climate,
   subcatchment_climate,prepare_climate_data_catalog}.py` (real code), leave old paths
   as re-export shims. Grep-derive the full importer inventory here. **Repoint the
   moved-module tests** (`test_extract_historical_climate.py`,
   `test_prepare_climate_data_catalog.py`, **`test_climate_forcing.py`**) to the new
   import paths in **this** commit — under Option B their assertions are unchanged (repo-1),
   so this is a pure import repoint, not a rewrite, and removes the shim from being
   load-bearing for the suite (repo-4). Gate: **full `pytest tests/`** (imports resolve),
   three `--dry-run`s.
2. `p32a: rewire wf3 script: paths + imports to climate_analysis` — repoint rule 3.02
   and rule 3.08 catalog-prep `script:` paths; rewrite live Python imports. **Diff of rule
   3.02 shows only the `script:` string changed** (C3 pin). Gate: `test_cli.py`,
   wf3 `--dry-run`, semantic-tree-diff clean on a wf3 slice.
3. `p32a: add wf1 raw-climate extraction (wrapper script) + model-parity module` — new
   `extract_climate_grid_wf1` rule with **`input: basin_nc = ancient(staticmaps.nc)`**
   (a declared rule-1.03 output — arch-1), `script:
   blueearth_cst/model/extract_climate_wf1.py` (the NEW dedicated wf1 wrapper, ext1-4 —
   NOT the moved module, whose wf3 `__main__` stays verbatim), writing
   `climate_historical/wf1_raw/extract_historical.nc` via `prep_historical_climate`
   called with the `staticmaps.nc`-derived `bbox=` (the shared function gains the
   optional keyword-only `bbox=` kwarg, wf3 call unchanged → wf3 output byte-identical,
   §5.1/§5.2). On the chirps branch the rule also declares the **`oro_nc =
   wf1_raw/orography.nc` sidecar output** (the wrapper relocates the shared
   function's sidecar to it — ext2-1, §5.2), and this commit lands the **parse-time
   eobs configuration guard** plus the wrapper's belt-and-braces assert (ext2-3,
   §5.2). Also lands `blueearth_cst/model/climate_parity.py` (the §5.2 named-callable
   parity transform incl. wf1-side PET — `meteo.precip`/`meteo.temp`/`meteo.pet`, never
   in the shared function) plus its **unit tests**: the bbox-tolerance assertion
   (per-edge ≤ 2×`model_resolution` vs the region bounds, ext1-5) and the
   **exactly-once correction tests** (era5 analytic + chirps sidecar-chaining, §5.2,
   ext1-1). Gate: wf1 `--dry-run` **on the default tracked config** (must be green —
   arch-1: `staticmaps.nc` has a producer, so the DAG builds), `test_cli.py`, the new
   unit tests green; confirm wf3 rule 3.02 output untouched.
4. `p32a: re-source wf1 subcatchment climate plots from raw grid (reproject-first)` — edit
   `plot_results.py` §4 to build `ds_clim` via
   `climate_forcing_by_subcatchment(model_parity_climate(ds_raw, dem_model, dem_forcing,
   pet_method), subcatchment_raster)` (§6.3 Option B through the §5.2 parity module,
   then the **unchanged** groupby); drop the `mod.forcing.data` path; load
   `staticmaps.nc["land_elevation"]` / `["subcatchment"]` wf1-side; update rule 1.10
   inputs and params (drop `inmaps` forcing, add the raw netCDF and — on the chirps
   branch — the **branch-resolved `oro_nc` sidecar input** (ext2-1, §5.2); add
   `data_sources` / `clim_source` params for the era5-branch orography fetch, §5.2;
   **leave the `script = ...` input
   label intact**, repo-5).
   Gate: wf1 `--dry-run`, `test_cli.py`, **full `pytest tests/`** (confirms
   `test_climate_forcing.py` still green — the function is unchanged, repo-1), wf1 e2e
   producing the new `clim_*` plots.
5. `p32a: characterized diff (ladder) + baseline record + manifest confirm` — land
   `dev/milestones/p32a/compare_climate_ladder.py` and run it: produce the side-by-side old/new
   plots, the persisted ladder states (`qa/l1_regrid_only.nc`, `qa/l2_parity.nc`) and
   aggregated tables A0/A1/A2, the component deltas (`A2 − A1`, `A2 − A0`, precip
   null-check, grid-level G vs `inmaps_historical.nc` — §5.5, ext1-3), and the
   fixture `wf1_raw` vs wf3 keyed-store `allclose` check (ext1-5, §5.2); record all in
   the `dev/milestones/p32a/` baseline note; run `check_baseline check` on the manifested wf1
   slice (expect no-op — the changed plots are unmanifested; record the knowing
   divergence from intake decision 4, repo-2). Gate: ladder recorded with every
   residual assigned to a named edge, G within §5.2 tolerances, manifested slice
   unchanged.
6. `p32a: delete transitional shims + MIGRATION note` — remove the old-path shims once no
   live citer remains; record the old→new module map. Gate: full `pytest tests/`, three
   `--dry-run`s, **and an execution-level check covering the two `script:` sites + the new
   wf1 rule** (the e2e `run_workflows.py` run, §9) — because dry-run does not execute
   `script:` bodies, a missed `script:` repoint only surfaces at execution (risk-6).

**MIGRATION map (recorded in commit 6 / milestone record):**
`experiment/extract_historical_climate.py` →
`climate_analysis/extract_historical_climate.py`;
`model/climate_forcing.py` → `climate_analysis/subcatchment_climate.py`
(function name `climate_forcing_by_subcatchment` **kept**, OQ-1);
`projections/prepare_climate_data_catalog.py` →
`climate_analysis/prepare_climate_data_catalog.py`.

## 9. Validation plan

**Reproducibility & validation gates (who checks each):**

- **Per-commit DAG gate** — `snakemake --dry-run` on all three Snakefiles +
  `pytest tests/test_cli.py` (the dry-run gate). *cst-architect verifies at each
  commit; delegated implementer runs.*
- **Full suite** — `pytest tests/` (incl. moved-module tests, wf1 build test where
  affordable). *python-engineer / model-builder on the implementation handoff.*
- **e2e** — `pixi run python scripts/run_workflows.py --config
  config/workflows/snake_config_model_test.yml` (all enabled workflows, fixed order)
  producing the new `clim_*` plots and a clean wf2/wf3. *model-builder.*
- **Value-identity (C2)** — `dev/scripts/semantic_tree_diff.py` pre/post on a wf2+wf3
  run: clean modulo nothing (only `script:`/import strings changed). *cst-architect
  reviews; implementer runs.*
- **Baseline manifest (C2)** — `dev/scripts/check_baseline.py check` on the wf1
  manifested slice: hydro/basin_area/precip PNGs + `output.csv` unchanged
  (no-op re-record for the changed-but-unmanifested `clim_*` plots). *model-validator
  confirms the manifested slice held; cst-architect records scope.*
- **Parity invariants (correctness of the re-source mechanism)** — the §5.2
  exactly-once unit tests (era5 analytic; chirps sidecar-chaining) and the
  bbox-tolerance unit test run in the full suite; the grid-level parity check G
  (parity grid vs `inmaps_historical.nc`, per-variable max-abs ≤ §5.2 tolerances)
  and the fixture `wf1_raw`-vs-keyed-store `allclose` check run in the commit-5
  ladder script. *python-engineer implements; model-validator reads G.*
- **Acceptance gate for the value change** — side-by-side old/new `clim_*` plots +
  the **ladder-decomposed** characterized delta record (§5.5: states A0/A1/A2 +
  grid-level G, persisted), recorded in the `dev/milestones/p32a/` baseline note. **User
  sign-off** at the milestone gate (visual QA); **stress-test-analyst /
  model-validator** computes and reads the ladder against the **edge-specific**
  acceptance logic (§5.5): the metric is the per-subcatchment monthly-mean delta per
  variable on each named ladder edge; the acceptance logic is — `A2 − A1` matches the
  named physical terms (T lapse term, EP PET response, P ≡ 0 exactly), `A2 − A0` and
  G ≈ 0 within §5.2 tolerances on era5 (on chirps: within the **pinned
  variable-specific hop tolerances** of the §5.5 defer-and-pin procedure — the gate
  is blocked until they are pinned on the first chirps basin, and unexplained
  excess over them fails it, ext2-2), every
  residual assigned to a named edge — and is **branch-aware** (era5 vs chirps;
  eobs is excluded by the §5.2 parse-time guard, ext2-3). No
  single "resolution-ratio" threshold.

**Acceptance criteria the outputs must meet (named for the validation handoffs):**

- C1: `grep -rE "WflowSbmModel|mod\.forcing|mod\.staticmaps"
  blueearth_cst/climate_analysis/` returns nothing.
- C2: semantic-tree-diff clean on wf2/wf3; manifested wf1 slice unchanged.
- C3: rule 3.02 diff shows only its `script:` string changed; `.guard_ok`/keyed-store
  paths byte-identical.
- Value change: ladder edges `A2 − A1` / `A2 − A0` / G consistent with their named
  causes per branch (§5.5); precip null-check exactly zero; no unexplained residual
  under model-parity; visual QA sign-off.
- Parity mechanism: exactly-once unit tests green (ext1-1); bbox per-edge within
  2×`model_resolution` of region bounds and fixture `wf1_raw` ≈ keyed-store (ext1-5).
- Contract edges (ext2-1, ext2-3): on a chirps config the dry-run DAG shows
  `wf1_raw/orography.nc` as a declared extraction-rule output **and** a declared
  rule-1.10 input (producer→consumer edge; no dirname-derived sidecar read in
  `plot_results.py`); `clim_historical: eobs` fails the wf1 dry-run at parse time
  with the named configuration error.

## 10. Open questions

- **OQ-1 — module/function rename (taste).** Rename `climate_forcing.py` →
  `subcatchment_climate.py`? **Recommend rename** (the file changes homes and loses its
  "forcing" meaning). The **function** name `climate_forcing_by_subcatchment` is **kept**
  (not renamed) — Option B leaves its signature and algorithm unchanged, so keeping the
  name minimises churn and keeps `test_climate_forcing.py` + the one importer byte-stable.
  A reviewer preferring `climate_by_zone` may still rename it, but that is churn without a
  mechanism win here.
- **OQ-2 — RESOLVED (ext1-2): PET for the raw grid.** The exact entry point is
  **`hydromt.model.processes.meteo.pet(ds, temp, dem_model, method=..., press_correction
  =True, ...)`** (`meteo.py:323`) — the very function the forcing build delegates to
  (`wflow_sbm.py:3707`) — verified against the installed hydromt 1.3.1: it takes plain
  xarray objects (no model), requires `press_msl`+`kin`(+`kout` for debruin), all of
  which the extraction already writes, and applies the model's pressure correction
  internally. Full call spec, units, correction order, and per-branch table in §5.2;
  locus (wf1-side only, in `model/climate_parity.py`) and method (the
  `setup_time_horizon.py:19-26` mapping: `debruin`/era5, `makkink`/eobs) are pinned.
  **The EP panel is retained unconditionally** — the v2 "reduced plot if infeasible"
  escape hatch is deleted (it would have been an unauthorized output change).
- **OQ-3 — project-level store (future).** Should the keyed climate store become a
  project-level, order-agnostic store feeding wf1+wf2+wf3 (removing the duplicate
  extraction)? Deferred to P3-2b/future; recorded so the duplicate is a known,
  bounded cost, not an oversight.
- **OQ-4 — manifest extension.** If a later decision adds the `clim_*` plots to wf1
  `rule all`, a manifest extension + re-record is needed. Out of scope for P3-2a.
- **OQ-5 — area-weighted aggregation.** Deferred accuracy improvement over the
  unweighted cell mean (`climate_forcing.py:44-47`); separate from the source-grid
  change.
- **OQ-6 — atomic cutover vs shim window.** §5.4/§6.4 selects the shim window;
  reviewer may prefer a single atomic transform commit.
- **OQ-7 — RESOLVED by selecting Option B.** v1's OQ-7 (the hydromt zonal-stats entry
  point for the polygon-zonal Option C) is **moot**: v2 selects Option B (reproject-first
  + the existing raster `groupby`, §6.3), which uses only `reproject_like` (already used
  in the same module) + the unchanged groupby — no unproven API, no polygon input. The
  polygon-zonal path and its zonal-stats/polygonization prerequisites are **deferred** to
  a future milestone (§6.3, OQ-8).
- **OQ-8 — model-independent subcatchment-zone source (future).** The aggregation still
  needs the subcatchment-zone raster from the built model (`staticmaps.nc`, §7 risk-2
  scope note). A genuinely model-free per-subcatchment-zone source (with a guaranteed
  id→station-index mapping) — which would also unlock the deferred polygon-zonal Option C
  — is out of P3-2a scope; a candidate for P3-2b / the standalone entry point.

## 11. Revision log

- **v1 (2026-07-24)** — initial draft for the G1 framing gate. Grounded against
  `AGENTS.md`, `dev/conventions/naming.md`, ADR 0002, R6 §8 (the DEFER fork),
  P3-1 §3a/§3d/§4 (keyed store + `.guard_ok`), the three Snakefiles,
  `check_baseline.py` TARGETS, and the lift-candidate modules. Key framing decisions:
  separate wf1 extraction (store-reuse blocked by pipeline order); model-independence
  defined as "takes no `WflowSbmModel`"; transitional-shim deprecation posture;
  manifest re-record confirmed near-empty (clim plots unmanifested).
  Corrected during drafting (advisor review) two mechanically load-bearing points:
  (1) the native-grid climate and model-grid subcatchment raster do **not**
  co-register, so a direct raster `groupby` is non-viable — §5.2/§6.3 reframed to
  select **polygon-zonal aggregation** (Option C), reproject-first (Option B) as
  fallback; (2) PET must be derived **wf1-side only** because `prep_historical_climate`
  is shared with wf3 rule 3.02 and adding `pet` there would break wf3 value-identity
  (OQ-2 constrained). Open questions OQ-1..OQ-7 flagged, notably the PET derivation
  (OQ-2) and the hydromt zonal-stats entry point (OQ-7) as pre-implementation items.
- **v2 (2026-07-24)** — revised after the internal lens panel (risk/architecture/repo-fit,
  all `revise`; blocking arch-1). Per-finding disposition in the review record (`climate-analysis-design-review-record.md`). Material changes:
  - **arch-1 (blocking):** the new wf1 extraction rule's declared input moves from
    `ancient(region.geojson)` — which has **no producer rule** in any Snakefile, so a fresh
    wf1 run and the commit-3 `test_cli.py` dry-run on the default config would raise
    `MissingInputException` — to `ancient(staticmaps.nc)`, a real rule-1.03 output. The
    shared `prep_historical_climate` gains an optional `bbox=` kwarg so the wf1 caller
    passes a `staticmaps.nc`-derived bbox while wf3's call (and output) stays byte-identical.
  - **Aggregation mechanism reversed (risk-4, arch-2, repo-1):** §6.3 now selects **Option B
    (reproject-first + the existing raster `groupby`)** over the v1 Option C (polygon-zonal).
    Option C required a per-subcatchment polygon GeoDataFrame that **does not exist** as an
    artifact and an OQ-7-unconfirmed hydromt zonal API; Option B needs only the existing
    subcatchment **raster**, keeps `climate_forcing_by_subcatchment`'s signature/algorithm
    unchanged (so `test_climate_forcing.py` — a **second live caller**, correcting v1's
    "none survive"/"assertions verbatim" claims — stays green with a pure import repoint,
    no rewrite), and yields a delta attributable to a known cause. OQ-7 is thereby resolved;
    the polygon path is deferred (OQ-8).
  - **§5.5 gate recalibrated (risk-1, risk-3, arch-3, risk-5):** the old path is
    lapse/pressure-corrected, PET-derived **model forcing** (`setup_time_horizon.py:56-70`),
    not regridded raw climate, so the "regridding-scale, exceeds-ratio ⇒ failure" heuristic
    is **dropped**. The delta is **component-decomposed** (regrid / temp lapse+pressure
    correction / PET-method), the wf1 path reproduces the model's correction + **PET method**
    (`debruin`) for parity, and the reasoning is **branch-aware** (era5 vs chirps —
    the no-reproject property is era5-only; the chirps branch already reprojects+lapse-corrects).
  - **§7 scoped honestly (risk-2):** "runnable without a built model" is claimed for
    **extraction only**; the aggregation still needs the built model's subcatchment-zone
    raster (C1-sense independence only). New OQ-8 tracks a model-free zone source.
  - Minors: arch-4 (named the new rule's `climate_nc`/params keys + the wf1-side caller for
    the `basin_nc` input), arch-5 (catalog module's sole consumer is wf3 rule 3.08, not wf2),
    repo-2 (manifest no-op named as a knowing divergence from intake decision 4), repo-3
    (`outlet_index.csv` removed from the manifested set), repo-4 (moved-module tests repointed
    in the move commit so the no-`__all__` shim is not load-bearing), repo-5 (rule 1.10's
    `script = ...` input label left intact), risk-6 (commit-6 adds an execution-level check
    for the `script:` repoints). No change to problem/scope/constraints/selected alternative
    beyond design detail — G1 framing stands.
- **v3 (2026-07-24)** — revised after external review round 1 (the review record,
  verdict `revise`). All five findings accepted; dispositions in the review record (`climate-analysis-design-review-record.md`). Material
  changes:
  - **ext1-1 (blocking):** §5.2 now carries an explicit **per-`clim_source`
    transformation pipeline table** — what the extraction has already applied per branch
    (era5: nothing; chirps: precip native, temp lapse-corrected to the chirps-grid MERIT
    DEM, press/kin/kout resampled-only; eobs: unverified, guarded), what the wf1-side
    parity module still applies, and a **net-correction ledger** (temp lapse net 1,
    pressure 1, precip 0). The chirps double-correction hazard is closed structurally:
    on that branch `dem_forcing` is **mandatorily** the extraction's orography sidecar,
    which `meteo.temp`'s MSL-shift algebra exactly inverts before re-correcting to the
    model DEM. Exactly-once is proven by two named tests: a synthetic unit test
    (era5 analytic + chirps sidecar-chaining, commit 3) and the fixture grid-level
    parity check G vs `inmaps_historical.nc` (commit 5).
  - **ext1-2 (blocking):** the parity callables are **resolved now, against the
    installed packages** (hydromt 1.3.1, hydromt_wflow 1.0.2):
    `hydromt.model.processes.meteo.precip` (`meteo.py:47`), `.temp` (`meteo.py:115`),
    `.pet` (`meteo.py:323`) — the same functions `setup_precip_forcing` /
    `setup_temp_pet_forcing` delegate to (`wflow_sbm.py:3310,3667,3707`) — with exact
    arguments, units, correction order, `dem_model = staticmaps.nc["land_elevation"]`
    (`naming.py:10`), per-branch `dem_forcing`, and the `setup_time_horizon.py:19-26`
    method mapping. OQ-2 is RESOLVED; the **EP panel is retained unconditionally** and
    the v2 "drop the panel if infeasible" escape hatch is deleted. New module
    `model/climate_parity.py` owns the transform.
  - **ext1-3 (major):** §5.5 defines an **operational comparison ladder** with named,
    persisted states (S0 = inmaps, S1 = raw extraction, S2 = regridded-only
    `qa/l1_regrid_only.nc`, S3 = parity `qa/l2_parity.nc`), components as deltas
    between adjacent aggregated states (`A2 − A1` correction component, exact precip
    null-check, `A2 − A0` sanctioned change, grid-level G), on-demand drill-down rungs,
    fixed-order interaction handling, branch preprocessing as a documented S1 property,
    and an attribution rule (every residual assigned to a named edge). Produced by
    `dev/milestones/p32a/compare_climate_ladder.py` (commit 5).
  - **ext1-4 (major):** the execution contract is pinned: a **dedicated wf1 wrapper
    script `blueearth_cst/model/extract_climate_wf1.py`** (imports the shared function,
    reads `sm.input.basin_nc`, passes `bbox=`) is the wf1 rule's `script:` target; the
    moved module's wf3 `__main__` block stays verbatim. §5.1 layout, §5.2 rule
    declaration, §5.3 contract, §8 commit 3, and the tests are reconciled to it.
  - **ext1-5 (minor):** the bbox derivation is defined (`ds_sm.raster.bounds` of
    `staticmaps.nc`), its relation to the region bounds explained (outward snapping to
    `model_resolution`), a per-edge ≤ 2×`model_resolution` fixture assertion added
    (commit 3), the `buffer=1`-source-cell insensitivity argument stated, and an
    empirical `wf1_raw` vs wf3 keyed-store `allclose` closure check added (commit 5).
  - The arch-1 bbox mechanism is upgraded from "preferred" to **selected** (the
    undeclared-file fallback (a) is dropped). No change to problem/scope/constraints/
    selected alternative — the framing (separate wf1-owned extraction, aggregation
    independent of the stored forcing artifact, shim-window migration) stands; **no G1
    return needed**.
- **v4 (2026-07-24)** — arbitration revision. The external round cap was reached
  after round 2 (the review record, verdict `revise`); the user arbitrated
  the three surviving findings (rulings final and binding, 2026-07-24) and this
  revision is **strictly confined** to them. Dispositions in the review record (`climate-analysis-design-review-record.md`).
  - **ext2-1 (major, accepted — fix required):** the chirps orography sidecar is
    promoted from an undeclared sibling file to a **declared output** of the wf1
    extraction rule (`oro_nc = climate_historical/wf1_raw/orography.nc`, a stable
    `clim_source`-independent path; the wrapper relocates the shared function's
    `{clim_source}_orography.nc` to it, shared module verbatim) and a **declared,
    branch-resolved input** of rule 1.10 (chirps: `sm.input.oro_nc`; era5: catalog
    fetch via params) — sibling-file discovery is eliminated. Both rules resolve
    the branch at DAG-parse time from the same config key, so Snakemake enforces
    sidecar existence and DEM/climate co-provenance on incremental runs.
    §5.1/§5.3 wrapper contract, §7, §8 commits 3–4, and §9 reconciled.
  - **ext2-2 (major, accepted — tolerance variant chosen; the direct-hop
    reference-state rung was explicitly not chosen and is not added):** §5.5
    replaces the subjective "small hop residual" chirps reading with a
    **defer-and-pin procedure**: chirps acceptance is blocked until
    variable-specific tolerances are pinned from the first chirps-basin ladder run
    (the pinned fixture, recorded state), each pinned value justified against a
    stated hop-mechanism bound (P exactly 0), and any later unexplained excess
    over the pinned tolerances **fails the gate**. §9 acceptance logic updated.
  - **ext2-3 (minor, accepted — exclude variant chosen):** eobs is scoped **out**
    of the P3-2a wf1 raw path: a parse-time configuration error in
    `Snakefile_model_creation` on `clim_historical: eobs` (plus a wrapper
    belt-and-braces assert) and a bounded support statement — **era5, chirps,
    chirps_global only**. No eobs pipeline is specified; the §5.2 eobs table row
    records the exclusion in place of the v3 "unverified + pre-implementation
    check" contract. wf3's use of the shared module is untouched (C2/C3).
  - Nothing else changed: problem/scope/constraints/selected alternative and all
    v1–v3 resolutions stand verbatim.
