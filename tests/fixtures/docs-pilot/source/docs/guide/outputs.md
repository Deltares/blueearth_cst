---
title: "Reading the outputs"
subtitle: "Find the generated climate series, model responses, and indicators"
---

Run products live under the configured `project_dir`. For production projects,
use a directory outside the toolbox checkout.

## The layout

```text
<project_dir>/
  config/                        configuration archives and run records
    analyze_climate/              latest WF0 run_record.yml and sources/
    build_model/                  latest WF1 run_record.yml and sources/
    analyze_projections/          latest WF2 run_record.yml and sources/
    _engine/                      execution inputs, locks and invocation history
  data/spatial/                  basin geometry, locations and maps
  data/climate/                  historical forcing and projection products
  models/hydrology/wflow/         model, historical run and evaluation
  scenarios/_engine/requests/<request_id>/  scenario generation settings
  scenarios/_engine/collections/<id>/
    collection.json              collection inventory
  scenarios/_engine/current.json  current scenario export identity and file inventory
  scenarios/                     stable copies of the current collection
    scenario_run_lookup.csv      generated series and their relationships
    perturbation_lookup.csv      monthly temperature and precipitation changes
    series/<run_id>.nc            generated climate series, e.g. run_01.nc
    config/                      copied creator configuration
    weathergenr/                 copied generator settings, products and diagnostics
  scenarios/_engine/data/<id>/   immutable collection, including original tables and series
  experiments/_engine/
    allocation.json             project-wide experiment numbering
    transactions/               experiment creation and recovery records
  experiments/<name>/
    _engine/simulation.json      simulation settings used
    hydrology/wflow/output/      retained model responses
    _engine/indicator_sets/<id>/
      indicators.json              indicator definitions and source records
      return_level_benchmark.json  scope of return-level checks
      coverage_evidence.json    per-series water-year block coverage
    _engine/indicator_sets/<id>/data/   immutable tables (same files as below)
    _engine/indicator_sets/names/<set-id>/
      current.json              committed publication of this set
    results/indicator_sets/<set-id>/    readable view of one indicator set
      indicators.csv            dictionary: units, method, policies, coverage
      group_members.csv         group_id,run_id (including singleton groups)
      group_metadata.csv        group_id,st_id,is_reference
      <token>_run_indicators.csv    indicator,location,run_id,value
      <token>_group_indicators.csv  indicator,location,group_id,value
  logs/<UTC timestamp>__<workflow scope>/
    summary.md                  execution summary and links
    console.log                 chronological launcher/child transcript
    wfN.log                     detailed messages grouped by rule/member
    wfN-benchmarks.md            compact timing and resource report
    wfN-benchmarks.tsv           precise individual-job measurements
    wf4-indicators.log          indicator details for this execution
    _parts/                     execution-owned working files
```

Two experiments can use the same generated climate series, but each experiment
keeps its own model responses and indicators. To relate an indicator to the
climate change that produced it, join a run table by `run_id` to `scenario_run_lookup.csv`,
then by `st_id` to `perturbation_lookup.csv`. For a group response surface, join the group table
by `group_id` directly to `group_metadata.csv`, then by `st_id` to `perturbation_lookup.csv`.
`group_members.csv` records the contributing runs. Reference groups have `is_reference=true`
and an empty `st_id`; retain them explicitly rather than losing them in an inner coordinate join.

Configuration archives are generated records. Edit the source configuration supplied to the runner;
the archived `sources/` files preserve what an execution used. WF3 and WF4 retain their configuration
inside the owning collection or experiment. The shared `experiments/_engine/` directory is reserved
for allocation and creation recovery; it is not an experiment.

Historical source-map fields live at
`data/climate/historical/<source_window>/tables/annual_climatology.nc`.
WF0 and WF1 share the same numerical producer. Diagnostic figures and the historical display
reference consume its sealed fields, so changing rendering or display classes does not repeat
annual-field preparation. Snakemake tracks the scientific inputs and implementation separately
from display dependencies.

Collections and indicator sets store immutable data under `_engine/`. The stable scenario export is
`scenarios/`. Indicator sets publish their readable tables at
`experiments/<name>/results/indicator_sets/<set-id>/`.

To inspect saved indicator sets, use `pixi run python scripts/inspect_indicators.py`.
Inspection checks saved hashes and table structure without rewriting the files.
Numerical rechecking also requires matching execution sources, inputs and recorded dependencies.
Integrity inspection does not make stale outputs reusable.

### Climate extent and basin cells

Historical climate stores retain a rectangular basin bounding box plus **one
source-grid cell on each side**. For ERA5's 0.25° grid, the margin is approximately
0.25° per side, with the exact extent determined by grid alignment.
This margin supports the current nearest-neighbour forcing reprojection; it does
not enlarge the modeled basin.

The adjacent `basin_cells.csv` lists cells whose footprints intersect the basin.
Only those cells, with equal weight, determine the weather generator's resampled
dates. Generated climate series retain the full buffered rectangle for subsequent
regridding. Existing stores and retained scenario collections are not cropped
in place; the smaller extent applies when climate inputs are regenerated.

## Check whether a run finished

Open `logs/<UTC timestamp>__<workflow scope>/summary.md`, linked by the runner's closing output.
The same execution folder holds `console.log`, `wf0.log`–`wf4.log` for invoked workflows,
and their `wfN-benchmarks.md` reports with raw `.tsv` measurements.
The summary reads the structured parent and child invocation records and lists elapsed time,
execution mode, observed outcomes, warning counts when available, and source/diagnostic links.
Standalone workflows also produce a summary. Repeated attempts retain separate execution folders.
An unfinished attempt or an older record without a measured duration stays explicitly incomplete
or unavailable. See [Run summaries and WF4 benchmarking](benchmarking.md) for interpretation.
Only WF4 records the additional process-resource timeline; the other workflows record invocation
durations without resource sampling. Successful execution does not establish scientific output completeness.

An incomplete run can leave logs and partial files.
Check `collection.json` or `indicators.json` in the corresponding `_engine/`
metadata directory before treating a collection or
indicator set as complete.

The runners record each attempt under `config/_engine/invocations/`,
including failed runs and dry runs. These records help diagnose a run; they do
not establish that its results are complete.

## Reading scenario exports

The stable `scenarios/` path mirrors every file from the selected hashed collection, including the lookup
CSVs, `series/`, `config/`, and `weathergenr/` trees. Successful WF3 generation or reuse refreshes the copies.
`scenarios/_engine/current.json` records the collection ID, revision, source marker, and exported file
checksums. WF4 continues consuming an explicitly selected immutable collection. Copied configuration and
provenance retain their original contents and references to the hashed collection.

The series copies require additional disk space. Files that already match are not recopied. Read the
export after workflow completion: updates replace individual files, not the whole set at once. Failed
generation or validation leaves the previous export intact; ordinary export I/O failures roll back
replaced files. Dry-runs do not refresh it. Only previously exported files are removed when absent from
the new collection; unrelated files remain, and untracked filename collisions are refused. Rerun WF3
to populate the stable paths in an existing project.

## Reading projection diagnostics

WF2 writes under `data/climate/projections/<ensemble>/`. Its `summary/` directory holds
the aligned annual and monthly change-factor CSVs. The separate `analysis/` directory
holds analysis-window tables, `coverage.csv`, and `availability.csv`.
The ensemble root's `_engine/analysis_record.json` describes the analysis.
Read the record for requested and used windows, exclusions, and numerical methods.

The figure suite lives in `plots/full_period/` and `plots/period_<a>_<b>/`, where `<a>-<b>`
identifies the actual analysis water years. Horizons sharing those years share a folder.
WF0–WF2 retain figure titles separately in `plots/figure_titles.md`, with matching figure keys in
`figure_titles.json` beside each caption JSON inventory. Titles are metadata for later independent
title/caption rendering controls; the default images still omit figure titles.

`plots/figure_captions.md` explains each written figure and unavailable products;
`_engine/figure_captions.json` inventories written PNGs, with paths relative to `plots/`.
Optional captioned copies
live under `plots/captioned/`. Missing eligible support produces a recorded reason, not
a placeholder PNG.

Generated plot filenames use lowercase snake_case with dataset, variable, plot context and spatial
scope fields, for example `cmip6_precip_temp_annual_change_cloud_basin_avg.png`. Internal settings,
the plot manifest and the analysis record share one `_engine/` at the ensemble root. User-facing provenance
remains with its data products in `summary/`.

### Which run produced these files

Runs launched through `scripts/run_workflows.py` or `scripts/run_workflow.py` keep
`_engine/execution/production.json` at the ensemble root. It names, for every current product file,
the invocation whose bytes are present, with that invocation's outcome, configuration snapshot under
`config/_engine/configuration-snapshots/analyze_projections/`, and `region.geojson` checksum.
Copies of each contributing invocation's `wf2.log` and `wf2-benchmarks.md`/`.tsv` sit beside it in
`_engine/execution/<invocation id>/`. They remain while that invocation still owns a file. Central
`logs/` folders and invocation records are never pruned by this.

- `state: complete` means the latest run that changed the product succeeded. `incomplete` means it
  failed after replacing some files: tables, figures or caches then come from different runs.
- A run that writes no product file, successful or failed, leaves the record untouched. A file is
  credited when the run changed its bytes or rewrote it identically (detected by write time under the
  project lock), so a partial plotting rerun credits only the figures and figure manifests it rewrote.
- `producer: null` marks bytes no launcher observed being written: products from before this record
  existed, direct Snakemake runs, or edits by hand. The record never credits them to a later run.
- Rules 2.01–2.02 run inside WF2 but write the shared `data/spatial/geoms/`. They appear in the copied
  log and are listed as `shared_steps`; their files are not product files.

## Moving WF2 results to another project

Export a completed WF2 product as one ZIP, then import it into a project on another machine:

```console
pixi run python scripts/transfer_results.py export --project-dir <source project> --output cmip6-results.zip
pixi run python scripts/transfer_results.py import cmip6-results.zip --project-dir <destination project>
```

The bundle holds the whole product (including `raw/`, `scalar/` and `_engine/`), the exact configuration
snapshots that produced it, the basin `region.geojson`, and a `manifest.json` listing every file's size and
SHA-256. It needs no source-machine paths, `_shared/` execution inputs, locks or `.snakemake/` state.
Source paths inside copied configuration are history, listed under `historical_path_references`.
Use `--ensemble <name>` when the project has no captured WF2 configuration naming the product.

Export refuses an `incomplete` product, missing required files, a `region.geojson` whose polygon differs
from the one recorded in `summary/provenance.json`, and provenance that matches no retained
configuration snapshot. Files whose bytes no longer match the production record are exported with
`producer: null`. Products without production records export as `legacy-unattributed`; their
configuration snapshot is the one whose digests match `summary/provenance.json`.

Import checks the whole bundle before writing: member names, sizes, checksums, the manifest's identity,
schema version and required files. It then compares the destination:

| Destination check | On mismatch |
|---|---|
| Basin polygon in `data/spatial/geoms/` | Refused, even with an equal bounding box; absent is `unverified` |
| WF2 ensemble, models, scenarios, members, variables, windows | Refused unless `--accept-differences` |
| Existing `data/climate/projections/<ensemble>/` | Refused unchanged; the same bundle is `already_imported` |

Settings are compared with `--config` when given, else with the destination's captured WF2
configuration. Accepted differences are recorded in the import record.

The product is staged in a hidden `.<ensemble>.import-*.staged` sibling and appears only after
verification, with one rename. A staged folder left by an interruption is never a result; the next
import removes it. Import adds `_engine/imported/<bundle id>/` with the manifest, `import_record.json`,
the configuration snapshots and region. It does not write the destination's `config/` archives,
`data/spatial/`, invocation records or logs; it only holds the project's execution lock under
`config/_engine/archive-transactions/`.

Imported results are for reading and interpretation. The runner has no record of producing them, so a
later WF2 run in the destination may recompute them, and WF2 does not need to be enabled to read them.
The bundle records toolbox commit, dirty state and lock hashes as provenance; they do not recreate the
source environment.

## Reading an indicator set

For post-processing, open the CSVs in `experiments/<name>/results/indicator_sets/<set-id>/`.
There is one folder per selected indicator set: `default` unless `project_config_simulate_system.yml`
selects others.

For example, a `default` set containing river discharge indicators looks like this:

```bash
experiments/stress_test_01/results/indicator_sets/default/
├── indicators.csv          # Indicator definitions, units and coverage
├── group_members.csv       # Runs included in each group
├── group_metadata.csv      # Climate perturbation and reference status of each group
├── q_run_indicators.csv    # Indicator values for each run and location
└── q_group_indicators.csv  # Indicator values for each group and location
```

Here, `q` means river discharge. Other selected variables have their own run and group tables.

Start with `indicators.csv`.
It has one row per indicator, describing:

- Its title, variable, input and output units, quantity, method and parameters.
- What each value represents (`grain`), how group results are calculated, the water-year start, and how days
  are assigned to blocks.
- Its partial-block, missingness and validity policies, references, and `block_coverage`.

`block_coverage` lists the blocks for every contributing run/location pair, including observed and
expected days and missing values.
This records coverage; it does not establish whether the sample is adequate.

For an indicator you name yourself, `equivalent_definition` identifies the existing calculation it uses.
The values are the same; they are reported under your indicator name.

Every successful WF4 invocation requesting indicators refreshes each selected set's folder, including
reuse of an existing set. `_engine/indicator_sets/names/<set-id>/current.json` records the committed
publication: the immutable indicator set, the captured set definition, the dictionary metadata and every
exported file's checksum. Sets publish independently: a failed set keeps its previous folder (the run
summary names such mixed-age exports) and the command exits nonzero. Editing only titles, descriptions
or authored citations refreshes `indicators.csv` without recomputing values.

Calculation identities include the selected calculation packages, their imported dependencies and shared
execution code. Adding or editing an unrelated indicator set leaves existing selections reusable; changing
a reused formula invalidates its dependent selections. Upstream implementation, logging and presentation
edits do not change calculation identity over unchanged retained responses. Resolved selections and
water-year settings, native response interpretation, grouping and stored value precision remain covered.
Snakemake continues to schedule the jobs; immutable publication guards still verify retained artifacts.

WF1 retains its full-precision historical `run_default/output.csv` for evaluation and table exports.
Changing evaluation settings does not rebuild thematic maps. Shared spatial preparation and WF1 model
updates declare their scientific helper modules as Snakemake inputs, so helper edits schedule their
producers without including logging or rendering modules.

Wait for workflow completion before reading an export: replacement is atomic per file, not across the
folder. Ordinary export I/O failures roll back the replaced files. Dry-runs and `simulations_only` do not
refresh exports. A folder is refreshed only when this set's committed publication owns it; an existing
folder with unknown files, or one claimed by another indicator set, is refused and left untouched. A
rerun restores managed files edited by hand; keep custom tables elsewhere.

Annual-total indicators (`aet`, `gwr`, `precip`) report **mm**: daily depths summed within each
water-year block and averaged with equal block weight. Partial first and last blocks are kept without
duration normalization and missing days contribute zero, so the value equals mean annual depth (mm/yr)
only when every block is complete and nonmissing; `block_coverage` shows whether it is.

Read `experiments/<name>/_engine/indicator_sets/<id>/indicators.json` before
its tables. It identifies the
indicators, the model responses they came from, and the files in the set.

Run tables contain one value per simulation for run-level indicators. Group tables contain one value
per design point for those indicators and pooled return levels. Both tables are written for every selected
variable; a scope without applicable indicators has a header-only table. Group membership remains explicit
even for one-member groups and selections without return levels.

Each run-level group summary is an equal-weight mean of the realization-level values, including annual
mean, extrema, quantiles, seven-day extrema, baseflow index, basin reductions and wet/dry-month means.
Unequal record lengths do not change realization weights. A NaN member propagates to the group summary;
infinite values are refused. Existing per-run reduction rules remain in force. Wet- and dry-month indicators
use the same month selected from the unperturbed reference and first native discharge location.

Return levels retain pooled estimation from member-local annual blocks, with the existing sample screening
and estimator validity rules. A singleton group uses those same rules, not a substitute run statistic.
The set records the members and reference location. Values are stored to four significant digits;
retain the underlying model responses for more detailed analysis.

### Opening an indicator table in Excel

Run IDs such as `run_01` and group IDs such as `group_01` are prefixed strings, so opening the CSV
normally keeps them as text.
When importing perturbation `st_id` values, explicitly preserve them as text in Excel and code.

When reading lookup tables in pandas, use `dtype=str` to preserve identifiers and
`keep_default_na=False` to retain the empty reference `st_id`.
Indicator `value` columns remain numeric.

Keep these identifier conventions in mind when joining tables:

- Each kind of ID uses a consistent number of digits within its collection or set.
- Groups are numbered independently of run count and selected indicators.
- Matching labels in different collections or indicator sets do not establish matching inputs or membership.

## Return levels and their limits

Return levels are fitted from groups of simulated years with a GEV distribution.
Read `return_level_validation` in `indicators.json` before using them. It identifies
the return periods and conditions assessed by the supplied benchmark;
`return_level_benchmark.json` holds that assessment. The benchmark covers a
limited set of synthetic cases. It does not establish accuracy for your basin,
and the results have no uncertainty intervals. Treat these return levels as
screening indicators, not design estimates.

The recorded GEV parameters use SciPy's `(c, loc, scale)` convention. The shape
parameter often written as ξ is `-c`; check this sign when comparing fits with
another tool. Each fit also reports `shape_coverage`, including how far its
estimated `c` falls outside the shapes tested in the benchmark. A fit outside
that range has not been assessed by this benchmark. Small departures are not
strong evidence on their own because shape estimates from short records are
uncertain.

Keep an indicator table together with its `indicators.json` and
`return_level_benchmark.json` when sharing or comparing return levels. The CSV
alone does not identify the estimator or its assessment.

## Are these outputs current?

Simulation checks that its requested scenario collection still matches the
current inputs. If the request is missing or out of date, the error names the
generation command to run. Simulation does not automatically choose a
collection. You can also name an existing collection explicitly in the
simulation settings file, `project_config_simulate_system.yml`.

The `_engine/simulation.json` record ties an experiment to its collection,
model, and simulation settings. Use a new experiment name when those inputs
change. If you change only the indicators, use `operations: [indicators]` to create a
new indicator set from retained model responses without rerunning simulation.

Simulation implementation identity follows the forcing preparation and Wflow
execution sources, including their scientific helpers. Python logging, figure
exports, generation implementation and configuration archive code are outside
that boundary; selected collection contents and resolved simulation inputs
remain checked separately. Snakemake continues to schedule jobs.

Changed scientific code, model inputs, settings, preparation, response requirements or relevant
environment require a new experiment. Adding temperature outputs changes the model's TOML digest
and also requires a new experiment. Incomplete experiments keep their strict input checks.
The Julia progress helper is part of the simulation identity because it invokes Wflow.

For a reproducible comparison, keep the whole project output directory, not
only the indicator CSVs. The model build settings used by HydroMT are recorded
in `models/hydrology/wflow/hydromt_build_config.yml`.

See [Running the workflows](running.md) to calculate different indicators
from retained responses.
