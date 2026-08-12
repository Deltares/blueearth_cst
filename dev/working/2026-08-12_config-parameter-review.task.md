# Task Brief — Configuration parameter review (planning only, no code)

### Context

Canonical ruleset: `AGENTS.md`. Inventory and problem statement already done:
`dev/working/parameter-placement.md` (DRAFT — its §5 rule is a proposal, not a
premise; do not treat it as decided).

- Surface is **three tiers**: toolbox (36 tracked config files), project (one
  `--configfile`, 55 leaf keys, 17 required), generated-into-`project_dir` (a
  record of a run, never an input).
- Defaults live in two places with no rule: 3 in
  `config/advanced_settings.yml`, 6 as Python `DEFAULT_*` backing config keys.
- Four inert or partly-inert parameters were found **by hand** in one session
  (WF2 `start_month_hyd_year`, `relax_priority`, `static_dir` in WF3, C34's
  `evaluate.model`). Nothing detects this class mechanically.
- Engine-native schemas (`config/defaults/*.yml`) are fixed by AGENTS.md's hard
  constraint — hydromt / wflow / weathergenr vocabulary is used verbatim. They
  are in scope to *describe*, out of scope to *restructure*.

### Goal

Answer five questions with evidence, so that a later design decision on
parameter organisation rests on a complete and checked picture rather than on
precedent. Produce findings and recommendations only.

### Non-goals

- **No code, config, schema or test changes.** Not a single edit to a runtime
  file. Recommendations are written down, not applied.
- No decision on where defaults should live — that is the *output* of this
  review, informed by Q4 and Q5, not an input.
- No restructuring proposal for engine-native templates.
- Not a design document. If the answers imply an architecture change, say so
  and stop; `design-document` owns that.

### Allowed scope

**Permitted (read only):** the whole repository, and the generated config tree
under `test_case/test_local/**` as evidence of tier 3.

**Write:** exactly one new file —
`dev/working/2026-08-12_config-parameter-review.md`. Update this brief's
`Progress` section as work advances.

**Forbidden:** every other path. Specifically no edits to `config/**`,
`blueearth_cst/**`, `Snakefile_*`, `tests/**`, `AGENTS.md`, or
`dev/working/parameter-placement.md` (the draft is an input; supersede it in
the review's own conclusions rather than editing it).

### Required changes (checklist)

The five questions, sharpened so each turns on one axis and they do not
overlap. Answer in order: Q5 must not reorganise parameters that Q1–Q3 would
delete, nor arrange names that Q4 would change.

1. **Q1 · Reach — which declared parameters never reach the computation?**
   Classify every one of the 55 project-config leaf keys and the
   `config/defaults/*.yml` keys into exactly one of:
   - **(a) never read** — no code reads the key at all;
   - **(b) read, unused** — bound to a variable or a rule `params:` and never
     consumed (`static_dir` in `Snakefile_climate_experiment:41`);
   - **(c) forwarded, dropped** — reaches a call boundary and is discarded
     before the arithmetic (WF2's water year until 2026-08-12; `relax_priority`
     at the `run_weather_generator` wrapper);
   - **(d) live** — a traced path to the consumer that uses it.
   For (a)–(c) give the exact point where the chain stops, as `file:line`.

2. **Q2 · Necessity — of the LIVE parameters, which should not be user-facing?**
   Distinct from Q1: a parameter can work perfectly and still not belong in a
   user's config. Flag each of:
   - only one value is ever valid (`static_dir` can only be `config`, because
     the fallbacks it feeds resolve to in-repo toolbox files);
   - never varied — identical across all shipped configs, and no stated reason
     a project would change it;
   - an implementation detail exposed by accident;
   - superseded by another key.
   Evidence: the set of values the key takes across the four `test_case/`
   configs, the template and `tests/snake_config_fixture.yml`, plus whether any
   other value is admissible.

3. **Q3 · Duplication — where is one concept declared more than once?**
   Four sub-cases, each needing both locations and **which one wins at
   runtime**:
   - the same value defined twice (`DEFAULT_ANCHOR` in
     `metrics_definition.py:18` and `climate_figures.py:120`);
   - one concept under two names, spellings or units (the water year was
     `start_month_hyd_year` as a month name and `year_start_month` as an
     integer);
   - a default in code *and* in config, so the effective value depends on which
     is consulted;
   - a value derivable from another already present (state the derivation).

4. **Q4 · Naming and documentation — are parameters named and described
   consistently?**
   Audit against `dev/reference/naming.md`, which already binds config keys
   (§2: snake_case keys, lowercase booleans, MUST for new keys; §7: rename a
   contract surface only with a migration note). Report separately:
   - **Where the convention is BROKEN** — cite the rule and the key.
   - **Where the convention is SILENT** — that is a finding in itself, and
     likely the larger one. It says nothing about units in names, about
     abbreviation (`realizations_num` vs `n_realizations` vs `RLZ_NUM` — three
     spellings of one quantity across the three tiers), or about word order
     (`start_month_hyd_year` vs `year_start_month`: same words, opposite
     order, different type).
   - **One concept, several names across tiers.** Every T2→T1 translation is a
     place a reader must hold two vocabularies. List them and say which are
     forced (an engine's own argument name, fixed by the hard constraint) and
     which are ours to fix.
   - **Units and types.** `gauge_snap_tolerance_m` and `river_uparea_km2` carry
     their unit; `resolution` (degrees) does not; `horizontime_climate` states
     neither quantity nor unit. Where is the unit — in the name, in a comment,
     or nowhere?
   - **Documentation completeness**, per key: is it in the template at all
     (`basin.hydrography` and `basin.basin_index` are not); is its default
     visible without reading Python (six are not); is required-vs-optional
     marked consistently; does its comment describe CURRENT behaviour?
     Several stale comments were found by hand this session — treat a comment
     that contradicts the code as a defect of the same class as an inert
     parameter, because both mislead a user who acts on them.

   **Beyond conformance — is the name any GOOD?** A key can satisfy every rule
   in naming.md and still fail to communicate. That half is judgement, not
   grep, and it is the half a user actually meets. Assess each name for:
   - **Word class matches value class.** A boolean should read as a predicate
     — `run_historical` says "run historical", which collides with
     `historical_window` and does not say it means "include the unperturbed
     baseline". A quantity should be a noun carrying its unit: `run_length` —
     length of what, measured in what? A path follows naming.md §5's
     `_path`/`_dir`.
   - **Is it English?** `horizontime_climate` is not a word, and states
     neither the quantity nor its unit.
   - **Abbreviation load, judged per case.** `st_num`, `clim_historical`,
     `wflow_outvars`, `nc_file_prefix`: which are established domain
     vocabulary a hydrologist reads fluently, and which are merely short?
     `st_` is pinned by naming.md §4 as a stable wildcard token and is not up
     for debate; the others are.
   - **Word order and qualifier position.** `clim_historical` puts the
     qualifier last while `historical_window` puts it first, for adjacent
     concepts. `data_sources` / `data_sources_climate` suffix a qualifier
     where two distinct names might read better.
   - **Does the leaf stand alone, or lean on its parent?** `max_per_basin`
     means nothing by itself; `automatic_subbasins.max_per_basin` is clear.
     Leaning on the path is legitimate, but it should be a consistent policy
     rather than an accident — and it fails the moment the key is quoted in a
     log line or an error message without its path.
   - **Does the name agree with what it points at?** `basin.gauge_points`
     resolves to a file called `output_locations.csv`. One of those two nouns
     is wrong.
   - **The reader test, which decides the rest:** would a hydrologist new to
     this toolbox correctly guess the key's meaning, type and unit from its
     name alone? Where the answer is no, record **what they would guess
     instead** — the plausible wrong reading is the finding, and is far more
     useful than "this name is unclear".

   Naming is a contract surface: any rename proposal must state its migration
   cost per naming.md §7, and must distinguish a name that is **wrong** from
   one that is merely **not what you would choose today**. Grandfathered names
   are worth breaking only for the first. Recommend, do not rename.

5. **Q5 · Organisation — is the hierarchy right for a user?**
   Answer each, with a recommendation:
   - Are three tiers the right tiers, or does the split hide something?
   - Within the project config, is `project` / `shared` / `workflows.*` the
     right axis — and is `shared` a coherent category or a leftovers bin?
   - Is nesting depth justified? `shared.basin.automatic_subbasins.max_per_basin`
     is four levels for one integer.
   - Is grouping by *kind*? `shared.basin` currently mixes basin definition,
     catalog bindings and delineation tolerances.
   - **The user-oriented test:** list every key a user must set or review to run
     a NEW BASIN. Is that set contiguous in the file, or scattered? If
     scattered, that is the finding — state the contiguous grouping that would
     replace it.
   - Where should a key's default be *visible* to the user? Give a
     recommendation and the argument against it.

6. **Rank every finding** by consequence, not by count: what could produce a
   wrong number, versus what is only untidy.

7. **Answer the P2 question the draft left open:** could Q1's classification be
   produced *mechanically* rather than by reading? Say whether a "declared keys
   ⊆ read keys" check is feasible against Snakemake's `params:` indirection,
   and if not, what the cheapest partial check would be. A judgement with
   reasons is an acceptable answer; an untested claim that it works is not.

### Progress

- [ ] Q1 reach classification
- [ ] Q2 necessity
- [ ] Q3 duplication
- [ ] Q4 naming + documentation audit
- [ ] Q5 organisation + user-oriented test
- [ ] Ranking and P2 feasibility

### Validation

No test suite applies — nothing executes. Validation is **evidence per claim**:

1. **Every claim carries `file:line`.** A parameter asserted inert names the
   line where its chain stops.
2. **Falsifier for each inertness claim.** The claim "X never reaches the
   computation" is disproved by exhibiting a call path from the config key to a
   consumer. State the search that would find one — e.g.
   `grep -rn "X" blueearth_cst/ Snakefile_*` plus the `sm.params.X` read — and
   report that it was run and returned nothing. Absence claims are the ones no
   amount of reading proves by itself; run the search that would refute them.
3. **Coverage is complete, not sampled.** All 55 project keys and all 14
   `DEFAULT_*` constants appear in the classification, including the
   uninteresting ones. A partial pass silently omits the inert parameter it was
   commissioned to find.
4. **Cross-check against the draft.** Where a conclusion contradicts
   `parameter-placement.md`, say so explicitly — the draft is unreviewed and
   may be wrong.

### Acceptance criteria

- All five questions answered, each finding evidenced and ranked by
  consequence.
- Complete coverage per Validation 3; no key silently skipped.
- Every recommendation states its cost and whether it breaks existing project
  configs.
- Open questions the review cannot settle are named as such, with what would
  settle them — not resolved by assertion.
- **Zero changes to any file outside `dev/working/`.** `git status` shows only
  the review document and this brief.

### Output requirements

One markdown file: `dev/working/2026-08-12_config-parameter-review.md`.

Structure: findings per question (Q1–Q5), then a single ranked recommendation
table — *finding · consequence · proposed action · cost · breaking?* — then
open questions.

No Results delta: nothing executes, so no results change.

### Task constraints

- Planning and review only. The first edit to a runtime file is a scope
  violation, however obvious the fix looks. Record it and move on.
- Do not treat `parameter-placement.md` §5 as decided; it is one input.
- Report assumptions and residual risk.

**Human gates**

- **Gate 1** — after Q1's classification, PAUSE. If it finds inert parameters
  beyond the four already known, the owner decides whether the review widens to
  cover them or records and continues.
- **Gate 2** — before writing Q5's recommendations, PAUSE for the owner to
  confirm whether breaking changes to the project-config schema may be
  proposed. (Latitude was granted for the earlier draft; confirm it still
  holds.)
- **Gate 3** — on completion, PAUSE. No follow-on implementation without an
  explicit new instruction.

---

## Appendix — parameter sets per file

A **dated snapshot, measured 2026-08-12**, reproduced here so the brief is
self-contained. It is the same measurement as the appendix in
`dev/working/parameter-placement.md`.

**Re-measure it; do not trust it.** Two copies of one inventory is precisely
the duplication Q3 exists to find, so the copy is only defensible if it is
checked: confirming these counts against the tree is the review's first act,
and any correction is itself a finding. `*` = required (`optional=False`).

### `config/advanced_settings.yml` — 5
`constraints.min_historical_years` · `defaults.julia_threads` ·
`defaults.seed` · `defaults.water_year_start` · `runtime.julia_version`

### Project config — `project` — 4
`project_dir`* · `static_dir`* · `data_sources`* · `data_sources_climate`*

### Project config — `shared` — 13
`basin.region`* · `basin.resolution` · `basin.gauge_points` ·
`basin.automatic_subbasins.max_per_basin` · `basin.gauge_snap_tolerance_m` ·
`basin.river_uparea_km2` · `basin.spatial_sources.{rivers,lulc,lai,soil}` ·
`historical_window.{starttime,endtime}`* · `clim_historical`*
— plus two optional keys the template does not document:
`basin.hydrography`, `basin.basin_index` (a Q2 candidate in themselves).

### Project config — `workflows.model_creation`
`enabled` · `model_build_config` · `waterbodies_config` · `wflow_outvars` ·
`observations_timeseries` · `simulation_window.{starttime,endtime}`*

### Project config — `workflows.climate_projections`
`enabled` · `clim_project`* · `models`* · `scenarios`* · `members`* ·
`variables`* · `historical_year_range`* · `future_horizons`* · `stats` ·
`save_grids` — `start_month_hyd_year` was retired 2026-08-12 and is now
refused at parse time.

### Project config — `workflows.climate_experiment`
`enabled` · `experiment_name` · `realizations_num` · `horizontime_climate`* ·
`run_length` · `run_historical` ·
`stress_test.{temp,precip}.{step_num, transient_change, mean.{min,max},
variance.{min,max}}` · `stress_test.{dry,wet}_spell_factor`

### `config/defaults/weathergen_config.yml` — 4 sections
`run_weather_generator` (2) · `generate_weather` (16 set + 6 injected) ·
`apply_climate_perturbations` (15) · `write_netcdf` (5). Sections are
weathergenr 1.2.0 function names, keys are their argument names — in scope to
describe, out of scope to restructure.

### `config/defaults/wflow_build_model.yml`, `wflow_update_waterbodies.yml`
hydromt `setup_*` blocks, verbatim in hydromt_wflow's schema. Same scope note.

### Data catalogs
`config/catalogs/` — `deltares_data.yml` (2596 lines), `deltares_data_linux.yml`
(1894), `cmip6_data.yml` (3919, **generated** by
`dev/scripts/generate_cmip6_catalog.py`), 2 archived ·
`tests/data/tests_data_catalog.yml` (112). Entry names are the contract; entry
bodies are hydromt's schema.

### Python `DEFAULT_*` — 14
Re-export tier 1: `DEFAULT_JULIA_THREADS`, `DEFAULT_SEED`,
`DEFAULT_WATER_YEAR_START`.
Back a config key — the Q3/Q1 candidates: `DEFAULT_SPELL_FACTOR`,
`DEFAULT_MAX_SUBBASINS_PER_BASIN`, `DEFAULT_GAUGE_SNAP_TOLERANCE_M`,
`DEFAULT_HYDROGRAPHY`, `DEFAULT_BASIN_INDEX`, `DEFAULT_STATS`.
Duplicated: `DEFAULT_ANCHOR` ×2 (`shared/metrics_definition.py:18`,
`climate_analysis/climate_figures.py:120`).
No config surface, correctly constants: `DEFAULT_DECIMALS`,
`DEFAULT_MIN_REFERENCE`, `DEFAULT_MAX_FLAGGED_MONTHS`.

### Process / build config — named for completeness, OUT OF SCOPE
`pixi.toml` · `pyproject.toml` · `Project.toml` · `Manifest.toml` ·
`.github/workflows/ci.yml` · `profiles/default/config.yaml` ·
`.testing-policy.yml` · `.git-workflow.yml` ·
`dev/reference/sealed-records.yml` ·
`dev/scripts/{stage_data,scaffold_extras}.yml`
· `config/templates/wflow_sbm.reference.toml` (reference only, nothing reads it)
· 5 archived single-workflow configs under `config/templates/archive/`.

### Tier 3 — generated per `project_dir`, a RECORD not an input
`config/runs/<workflow>/<digest>/{source.yml, effective.yml,
referenced-files.json, files/**}` (content-addressed run snapshots) ·
`config/catalogs/*` · `config/templates/*` ·
`experiments/<id>/{experiment.yml, model_reference.yml,
snake_config_climate_experiment.yml, catalogs/*, runs/**}` ·
`experiments/<id>/climate/weathergenr/config/weathergen_config.yml` ·
`models/hydrology/wflow/{wflow_sbm.toml, config/*}`

Tier 3 is in scope for **one** question only: does anything read a tier-3 file
back as an INPUT? If so, that is a Q1 finding of a different kind — a record
being used as configuration.
