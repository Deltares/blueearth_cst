# Changelog

Personal fork of `Deltares/blueearth_cst`. Release history follows
[Keep a Changelog](https://keepachangelog.com/) loosely.

Releases follow the [manual toolbox versioning policy](dev/reference/versioning.md):
Git release tags are authoritative; milestone tags are development checkpoints.
Milestone detail lives in `dev/roadmap.md` and `dev/records/milestones/`.

## [Unreleased]

- Write annual anomaly multi-panel figures directly under `plots/`; use `_page_01` suffixes only
  when the panel limit requires multiple figures, and retire managed PNGs from the older nested layout.

- Keep historical climate comparison plots and unique source plots by default; make duplicate
  single-source views, model-forcing figures and individual location figures opt-in through YAML.
  Preserve spatial maps, numerical diagnostics, simulation responses and performance tables.

- Retire managed WF0 annual anomaly pages under both current and legacy filenames, preserving unrelated files.
- Normalize Pixi boolean environment values when borrowing a session runtime.
- Escape unsupported characters in redirected console output while preserving the original UTF-8 log.

- Select stochastic or delta-change generation through schema-3 method sections and the workflow runner.
  Preserve legacy stochastic settings and explicit delta collection selection; preview delta inputs without
  publishing, record captured generation inputs, and reject unsupported flags or incomplete output reuse.

- Standardize WF0–WF4 console openings with literal command display, labelled plans, conditional
  input checks and clear execution/dry-run boundaries, including compact up-to-date output.

- Reduce repeated WF2 null-band calculations, trajectory work and verified irregular-grid reads;
  validate source compatibility and analysis records earlier. Reduce caption measurement overhead
  while retaining the existing PNG compression.

- Serialize NumPy-valued NetCDF metadata in WF2 analysis records without dropping attributes or
  relaxing strict non-finite JSON checks.

- Reorder unsorted CMIP6 time blocks before temporal slicing, retaining timestamp/value pairs and
  all records for the existing duplicate and coverage checks; report the reordering in fetch logs.

- Expand WF2 run settings with analysis variables, methods, aggregation and resolved ensemble counts;
  cap displayed model names at three while reporting omitted and total counts.

- Preserve source-specific scalar measurement heights when merging WF2 monthly inputs, allowing
  2 m temperature/humidity and 10 m wind on matching grids without overriding coordinate conflicts.

- Ignore Python checkout line endings in historical map identities; existing preparations require
  one regeneration after this identity change. Data and environment lock hashes remain byte-exact.

- Handle a single catalog path correctly when hashing WF0 annual spatial anomaly inputs.

- Validate named simulation indicator settings before upstream workflows start, while keeping forcing
  validation and experiment admission at the first forcing consumer.

- Preserve implicit legacy climate candidates when staging execution configuration with public forcing keys.

- Keep native shared-producer declarations aligned for an unambiguous historical climate source,
  without applying draft model-forcing adjustments to comparison workflows.

- Accept native CHIRPS `mm d**-1` precipitation units when computing historical climate indices.

- Admit historical evaluation and named-run retention settings during all-workflow experiment preflight.

- Preserve shared axes in the weather generator's conditional precipitation-correlation diagnostic,
  avoiding a plot-export assertion that stopped scenario generation after evaluation.

- Remove workflow log/benchmark gather rules from WF0–WF4. Owned launchers finalize diagnostics,
  including partial failures; direct Snakemake execution no longer gathers diagnostic reports.

- Scaffold external cases with named peer trials (`default` by default), portable schema-3 paths,
  explicit `--from-trial` configuration copying, and a minimal case README inventory.
  Existing split-layout configurations remain runnable; migration is copy-and-verify, not automatic.

- Add opt-in monthly liquid-water dew point and explicit 10 m scalar/component wind analysis,
  with strict native metadata admission, cell-first derivation, absolute state changes and captions.
  Controlled verification does not qualify live input availability.

- Add opt-in monthly CMIP6 PET analysis using explicit HydroMT Makkink or de Bruin methods,
  native-cell derivation before buffered-bbox averaging, strict source admission, calendar-aware
  totals, inspectable provenance and three PET figure families. Real additional-input qualification
  remains separate from controlled mechanism verification.

- Reject incomplete native grid support, shifted grids and incorrect units before publication or retry
  reuse, using model-derived expectations while preserving portable retained readers.

- Add independent optional native Wflow NetCDF grids for historical and scenario simulations,
  preserving native layers and scalar results, with frozen requests, retained integrity evidence,
  conservative disk estimates and paired CSV/NetCDF retry recovery.

- Add optional observed catchment water-balance screening against prepared historical forcing, with
  explicit daily-mean UTC support declarations, complete matched years and retained diagnostic tables.

- Add optional historical rain-gauge comparisons with native-cell sampling, explicit UTC daily labels,
  common observed-only dates, strict complete-period totals, inspectable tables and station figures.

- Consolidate public forcing selection in project-level `climate.forcing.precip_source` and
  `temp_source`. Infer precipitation only from a sole valid candidate when required; comparison-only
  analysis needs no selection. Accept legacy keys with notices, reject alias collisions and invalid
  supplied values, preserve scientific serialization and show resolved choices before execution.

- Add selected historical forcing recipes for companion climate, source elevation, de Bruin/Makkink
  PET and independent temperature/pressure corrections. Computed recipes extend through scenario
  generation and retained simulation preparation; supplied reference PET is historical-only.
  Separate extraction identity from preparation settings and retain version-2 preparation compatibility.
  No new data products are included; alternative-source qualification remains separate.

- Fix CHIRPS historical model forcing by passing the extraction orography sidecar into its
  generated catalog and using that elevation reference for temperature downscaling. This avoids
  a missing-sidecar error and applying the extraction lapse correction twice; ERA5 is unchanged.
- Add a standalone monthly delta-change scenario producer with explicit projection-member provenance,
  historical and neutral controls, and WF4 simulation and indicator grouping. Stochastic generation remains
  independent of projection selection.

- Bundle the FAO-style daily mean-temperature counts into historical climate indices: days strictly
  below 0°C and above a configurable threshold (default 25°C), with basin/grid annual and seasonal
  products. These are explicitly distinguished from minimum-temperature frost days and heatwaves.

- Add opt-in historical climate indices for basin series and native grids, with retained references,
  event catalogues, annual/seasonal NetCDFs and figures. Temperature exceedance percentages use
  reference-year bootstrap correction; WSDI uses fixed thresholds to match its event catalogue.
  Existing diagnostic tables and formulas remain unchanged.

- Use a shared style and consistent panel layout for WF3 weather figures, preserving
  their filenames and scientific results. Pin the required `weathergenr` 2.1.0 implementation by
  published commit and verify its version and source during installation.

- Bound and retry Windows socket-pair setup in the owned Snakemake launcher, preventing an
  indefinite scheduler wait when a local connection fails during job completion. Persistent
  timeouts report an error; completed outputs remain available for incomplete-job resume.

- Fix --unlock being rejected by the WF0–WF2 source-capture guard when forwarded through
  the all-workflow runner.

- Fix project rebuilds leaving stale WF3/WF4 archive journals that prevented scenario generation
  or experiment publication with `archive terminal state is corrupt: committed`.

- Add `run_workflows.py --rebuild` to permanently reset generated outputs before rerunning enabled
  workflows. It asks for confirmation, supports `--yes` for automation, and offers a read-only erasure
  preview with `--rebuild -- --dry-run`. Source inputs, unrelated paths, logs and history are preserved.

- Clarify WF3/WF4 console summaries, show the weather template notice once per generation invocation,
  and use stable scenario export paths in console output while keeping engine paths in detailed logs.

- Show the user-facing project config as the first Paths row in every workflow's console header.

- Generate concise case instructions with copy-paste commands, incomplete-job reruns and unlock,
  and a configuration summary at the end.

- Add portable project schema 3: file paths resolve beside their declaring YAML, and `toolbox:`
  locators select bundled resources. Case scaffolds and shipped seeds use the new contract.
  Schema 2 retains its existing path semantics; see [migration guidance](docs/technical-manual/configuration.md).

- Support flat inline WF4 indicator-set configuration, inherited parameter defaults and per-indicator
  `enabled` flags. Existing selectors and set files remain supported. Historical Q90 rules can use
  `baseline_quantile: {}` with the project's WF1 model and automatically derived source provenance.

- Defer historical Qcrit preparation until WF4 after enabled upstream workflows. Combined dry-runs
  report missing WF1-dependent Q90 preparation as pending; direct WF4 reports actionable missing
  historical CSV/TOML prerequisites. Distinguish absent indicator parameters from response variables.
  Omitted baseline-quantile years and `years: all` select all historical calendar years, retaining
  selection mode, discovered/requested years and per-location exclusions in calculation provenance.
- Add the optional `eflow-erfa` indicator set: 35 annual IHA-style flow metrics, each summarized
  by median and IQR, ported from `eflowrisk` defaults. Select it through WF4 `indicator_sets`.
  Set identifiers now also accept hyphens; existing underscore identifiers remain valid.

## [v0.4.1] — 2026-10-12

This release uses a patch number despite new features and
incompatible output paths since `v0.4.0`. Read the migration notes before reusing existing projects.
Git release tags remain authoritative.

### Breaking and migration

- Configuration archives move from `config/runs/` to `config/`; experiment allocation and recovery
  metadata move from `config/experiments/` to `experiments/_engine/`. Existing output trees are not
  migrated or supported through path aliases. Use a fresh output root; retain older trees for reference.
- Fresh immutable scenario data live under `scenarios/_engine/data/<id>/`; fresh metric tables live
  under `experiments/<name>/_engine/metric_sets/<id>/data/`. Readers preserve completed legacy layouts.
  Use the source manifests for immutable paths and the stable exports described in the
  [output guide](docs/guide/outputs.md) for post-processing.
- Generation, simulation and metric identities track imported code. These changes can invalidate
  reuse; create a new experiment when the runner refuses a changed simulation identity.
- WF2 diagnostic tables now use complete water years within the configured reference and future
  windows, replacing fixed 30-year windows and historical fallbacks. Seasonal eligibility requires
  90% of each owned window, rounded up, with at least three seasons. The analysis record now stores
  separate reference/future thresholds per window. Rerun analysis and rendering; diagnostic values
  and period folders can change. Overlay change-factor arithmetic is unchanged. See the
  [WF2 reference](docs/site/toolbox-reference/workflow-analyze-projections.qmd).

### Added

- Stable scenario exports mirror the selected collection at `scenarios/`; stable metric CSV exports
  publish the selected set at `experiments/<name>/results/metric_sets/`. Engine inventories identify
  exported files and checksums. Exports refresh after successful generation or indicator requests,
  including reuse; read them after completion and keep custom files under distinct names.
- Monthly empirical percentile calendars with reference-support reporting, clearer precipitation
  anomalies, and improved station-extreme and signature figures.
- Runner summaries report warning counts and write command receipts for supported reruns.

### Fixed

- Climate and forcing maps fit basin extents and retain readable spatial keys after layout.
- Console settings and command options wrap within the terminal width; catalog chatter is captured.
- Case scaffolding preserves requested destinations, and summary failures are reported explicitly.
- Seasonal diagnostics scale coverage to each configured analysis window rather than requiring
  27 seasons for every window. Shorter valid windows can now produce previously unavailable results.

### Known limitations

- The full five-workflow test was incomplete for `0.4.1` because of slow P-drive reads. The fresh
  run completed WF0 and WF1 and stopped during WF2.
- Diagnostic changes affect scientific results; no whole-tree numerical equivalence is claimed.
  The baseline manifest remains stale after the P-drive/GlobCover input switch and cannot qualify
  current numerical results without fresh baseline generation and accepted re-recording.
- Linux/Docker full-pipeline scientific replication remains unqualified. Local all-five execution
  requires the data mirror, Julia, `weathergenr` and CMIP6 network access.
- Stable exports duplicate retained data and replace files individually; read after workflow completion.

## [v0.4.0] — 2026-10-10

Pre-1.0 minor release with incompatible experiment identities and output contracts since `v0.3.0`.
The Git release tag defines the version.

### Breaking

- Scenario runs and perturbation groups now have separate, prefixed string IDs (`run_01`, `group_01`).
  Updated collection and metric schemas refuse predecessor numeric-ID collections. Regenerate these
  collections and simulations; update consumers using the
  [output contract and migration guidance](docs/guide/outputs.md).
- New cases use the case root directly as `project_dir`, without an `active/` suffix. Existing explicitly
  configured `active/` roots remain valid; changing scaffolds does not relocate retained outputs.
- Scenario-collection, simulation and metric-set ids all change. Each identity
  hashes the repository code its stage imports, and removing the pre-release v1
  records edited shared modules on all three paths; the relocation fix below
  moves the metric-set id once more. The climate-store identity does not change. Existing collections, experiments and metric sets stay
  readable, but none is reused by a fresh run:
  - WF3 generates a new collection instead of reusing the retained one.
  - Re-simulating an existing experiment is refused; use a new experiment name.
  - A metrics-only run over a retained experiment publishes a new metric set
    with identical values.

### Added

- Numbered experiment allocation with `--new-experiment`, immutable request bindings and guarded
  continuation. A plain invocation selects the highest existing experiment in its family, including
  a completed one; only explicit creation advances the number. Changed scenario requests are refused
  before producers run. Compatible interrupted simulation batches retain their native checkpoints.
- Canonical WF0 climate diagnostics and source-comparison figures; expanded WF2 ensemble coverage,
  analysis windows, seasonal reductions and variability benchmarks.
- Shared historical map references and chart preparation across WF0/WF1, with standardized forcing,
  hydrograph, model-performance and hydrological-summary figures.
- Each metric set retains its return-level evidence as
  `_engine/metric_sets/<id>/return_level_evidence.json`, SHA-referenced from
  the manifest and verified on read. For every fitted return level it records
  the usable block counts per member against the required minimum, the fitted
  parameters, fit and shape coverage, and the policies applied. Sets published
  without it stay readable.

### Changed

- WF2 reads the public CMIP6 bucket anonymously. With no token, each fetch
  process spent about 19 s searching for Google credentials first.
- The metric-set reader accepts only `metrics.json` as a set's engine marker.

### Fixed

- WF2 monthly-factor figures now honor the configured water year. Geographic frames, map legends,
  captions and figure export preserve their intended extent and layout.
- Wflow forcing checks honor the retained state's timestamp; region preparation reports outlet-search
  and tiny-raster failures explicitly.
- A project tree copied to another root reuses its ready scenario collection,
  simulation and metric set instead of refusing them. The installed
  weather-generator input and each metric plan record absolute paths under
  the original root, and reuse compared those paths against the new root.

### Removed

- The v1 -> v2 project-config migrator (`scripts/migrate_project_config.py`),
  its mapping (`config/migrations/v1_to_v2.yml`) and its guide. Migrate a
  pre-R14 project config with release `v0.3.0`, which still ships all three; the
  loader's refusals now say so. `ruamel.yaml` is no longer a declared dependency.
- Support for the pre-release v1 records: the `config/simulation.json`
  simulation record, the v1 response inventory, `scenario-collection/1`, and
  `metric-set/1`. No release wrote them; they existed only between 2026-09-11
  and 2026-09-22. Such an experiment is now refused by name; simulate it again
  under a new experiment name.

### Migration and known limitations

- Use the [owned runners](README.md) for generation and simulation. To change a scenario request,
  explicitly create a new experiment; an ordinary invocation does not silently replace completed
  or interrupted work. Recovery of reservation metadata does not repair damaged native outputs.
- Retain older project trees before regenerating incompatible collections or simulations. Existing
  explicit project roots remain supported, including paths ending in `active/`.
- Plot and diagnostic additions do not establish cross-platform numerical equivalence. Linux/Docker
  full-pipeline scientific replication remains unqualified, and sample catalogs can require local data.
- The baseline manifest predates the P-drive/GlobCover input change. It must be regenerated and
  re-recorded from the revised baseline configuration before use for numerical checks.

## [v0.3.0] — 2026-10-01

Pre-1.0 minor release with incompatible user interfaces, output contracts,
and scientific calculations since `v0.2.0-alpha`. The Git release tag defines the version.

### Breaking

- Reorganized the shipped code into `blueearth_cst/`, consolidated configuration
  and project output layouts, and renamed workflow rules. Existing scripts,
  rule targets, and output consumers need migration.
- Split project configuration into a project file and per-workflow files, with
  closed workflow stanzas and shared keys owned by the project file's `basin`,
  `climate`, and `model` sections or top level. See the
  [project-config migration guide](docs/site/guide/migrating-project-config.qmd).
- Reworked projections analysis around monthly GCM series. `save_grids` became `save_gridded`; variable
  declarations now specify canonical names and change semantics. Missing catalog models fail at DAG
  construction; unavailable scenario/member combinations are reported as skips. See the R8 migration record
  (private-checkout-only citation: `dev/records/milestones/r08/migration-wf2.md`).
- Separated scenario generation (WF3) from system simulation and retained-response
  metrics (WF4). Scenario collections, explicit run IDs, immutable metric sets,
  and their output contracts replace the predecessor experiment layout. Use the
  owned workflow runners described in [README.md](README.md).
- Result locations now use the registry `wflow_id`; the baseline outlet changed
  from `101` to `1010`. Consumers must use registry identifiers.

### Added

- Model-independent historical climate analysis (WF0), forcing-readiness
  reporting, model interchange contracts, and expanded practitioner documentation.
- Projection annual and monthly change-factor tables, requested-combination
  status reporting, provenance and limitation reports, and reusable series
  caching with separate fetch and reduction steps.
- Durable scenario collections with generation-code identity, simulation batch
  recovery, batching calibration, and retained-response metric validation.
- Cross-platform CI, focused validation checks, and the manual toolbox release
  policy and checklist.
- A single-workflow selector (`scripts/run_workflow.py`) for all five workflows,
  delegating generation and simulation to their owned runners; external basin
  case scaffolding through `scripts/create_case.py`.

### Changed

- Projection spatial means use spherical cell-area weights; annual aggregation
  uses month lengths in each model's calendar. Reduction no longer rounds to
  two decimals; default statistics are mean, median, and standard deviation,
  with labelled, opt-in tail quantiles. Figures show individual combinations.
- Scenario identity follows generation code; downstream simulation changes do
  not invalidate generation. Simulation retains one discharge series per model
  cell and stages completed batch members for reuse after interruption.
- WF3 generation-input snapshots now live under the scenario collection's
  `_engine` directory.

### Fixed

- Projection reference windows include the final complete hydrological year
  (`1990–2010` includes 21 years). Model calendars are read from their stores,
  and reduction-kernel identity now invalidates stale cached arithmetic.
- The unperturbed scenario now passes through the same perturbation step as
  other scenarios. In the recorded baseline, affected low-flow results decreased
  by up to about 90%; the accepted baseline was re-recorded on 2026-09-25.
- Restored workflow log and benchmark gathering, corrected checkpoint lookups,
  and repaired Windows console capture during simulation preparation.
- Serialized netCDF access during staging to avoid the observed lock-order
  stall, and corrected projection flagged-month dataset identities.

### Migration

- Migrate older project configuration with
  `pixi run python scripts/migrate_project_config.py <project-file> --dry-run`,
  review the proposed changes, then run without `--dry-run`. See the
  [current migration guide](docs/site/guide/migrating-project-config.qmd) for
  backup behavior and cases the migrator refuses.
- Update workflow commands using [README.md](README.md), and update consumers
  against the [current output structure](docs/guide/outputs.md).
  Configuration migration alone does not migrate predecessor output trees;
  retain the old tree and regenerate into a separate project directory.
- Recompute affected projections and scenario/simulation results under the
  current contracts. Changes to weighting, calendars, reference windows, and
  baseline perturbation mean numerical parity with the preceding release is
  not expected. The measured low-flow change above describes the recorded
  baseline, not a bound for other basins.

### Known limitations

- Docker and Linux end-to-end scientific replication remain unqualified;
  cross-platform CI does not establish full-pipeline numerical equivalence.
- A portable sample-data bundle is not shipped. Existing seed catalogs can
  depend on machine-specific data locations; configure accessible data before
  running a new basin. See [installation](docs/site/setup/install.md).
- WF0 provides historical characterization and forcing-readiness reporting;
  the planned forcing-selection evaluation layer remains future work.

## [v0.2.0-alpha] — 2026-05-09

Completed the foundation work and designed the workflow refactor.
The minor version increase covers incompatible library upgrades, repository changes,
and tests needed for the refactor.

### Phase 1 (Foundation, completed)

- **M01 — Replication baseline** (`m01-replication`): all three
  Snakemake workflows replicated end-to-end on the test config;
  baseline manifest at `dev/baseline/manifest.json`;
  `dev/scripts/check_baseline.py` with `record` / `check` subcommands.
- **M02 — Pixi environment** (`m02-pixi`): Python and R managed through Pixi,
  with Julia managed through juliaup, replacing conda and manual setup.
- **M02b — Library upgrades** (`m02b-upgrades`): hydromt 0.x → 1.3,
  hydromt_wflow 0.x → 1.0.2, Wflow.jl 0.7 → 1.0.2, Python stack caps
  lifted (numpy 2.x, xarray latest). Baseline regenerated with the numerical changes documented.
- **M02c — Test coverage** (`m02c-tests`): 32 new unit tests across 4
  `src/` modules plus 2 strict xfails for documented bugs (hydromt
  `to_yml` preprocess strip; `extract_climate_grid` silent truncation).
  Established the `sys.modules.setdefault` mocking pattern.

### Phase 2 (Refactor) — designed, not yet executed

- **R1 design** (`dev/records/milestones/r01/`): per-workflow modularity contracts.
  Sectioned config schema (`project` / `shared` / `workflows.<name>`),
  per-workflow contract docs, atomic migration plan including
  `src/` script updates and baseline-snapshot policy.
- **R2 design** (`dev/records/milestones/r02/`): naming conventions
  guide (snake_case, lowercase acronyms, `_path` canonical, suffix
  vocabulary split between paths and data objects, domain-identifier
  exceptions, and names requiring migration notes).

### Toolchain

- **Python 3.12 → 3.14.4**. The deprecated `Path(fn, resolve_path=True)`
  argument in `src/prepare_climate_data_catalog.py` was fixed before the upgrade.
- **Snakemake stays at 9.6.2**. Bump to 9.20 deferred — snakemake 9.7+
  requires `packaging<26` but the conda env pulls in `packaging==26.2`
  through another dependency.
- `datrie` pin removed (was redundant; modern snakemake doesn't need it
  as a hard conda dep).

### Repo structure

- Roadmap restructured into Phase 1 (completed) / Phase 2 (active)
  sections with clear visual break.
- `dev/records/milestones/phase-1/{m01,m02,m02b,m02c}/` for sealed foundation milestones.
- `dev/records/milestones/r01/`, `dev/records/milestones/r02/` for active Phase 2 milestones.
- `dev/reference/` reserved for R2 output.
- Stale forward-looking M3-M6 references renamed to R3-R6 across docs,
  tests, and code comments.
- New tag prefix convention: `r##-<topic>` for Phase 2 (e.g.
  `r03-model-builder`); `m##-<topic>` preserved for sealed Phase 1.

### Testing

- Suite total: 45 passed, 4 xfailed. Unchanged through the entire
  toolchain bump and repo restructure (validates that the moves and
  text updates did not affect the tested behavior).

### Documentation

- README updated to reflect fork status, Python 3.14, Julia 1.11.x via
  juliaup, deferred Docker, and pointers to `dev/install.md` and
  `dev/roadmap.md`.
- New `CHANGELOG.md` (this file).

### Known issues / followups

- Test `xfail`s tracked in `dev/tasks/`:
  - hydromt `to_yml` preprocess strip — upstream hydromt issue.
  - `extract_climate_grid` silent truncation when staged source < requested window — fix in R3.
- Snakemake bump to 9.20 deferred until upstream relaxes the
  `packaging<26` constraint.
- Linux / Docker validation deferred (see "Deferred: Linux replication"
  in `dev/roadmap.md`).

## [v0.1.0-alpha] — base

Upstream starting point. Forked from `Deltares/blueearth_cst` at this
version. Branch `base/v0.1.0-alpha` preserves the exact starting
commit. All Phase 1 work in this fork builds on top of this baseline.
