# Run identity, reuse, reservations and batching

Last updated: 2026-11-17

Use this reference before changing what identifies a scenario collection, a WF4 simulation or an indicator
set, before editing code inside a producer's pinned closure, and when changing experiment reservation or
WF4 batch sizing. The [retained-handoffs guide](../../docs/technical-manual/retained-handoffs.md) explains
saved products and reuse to users; the [architecture reference](toolbox-architecture.md) owns workflow
boundaries. The modules named below own exact schemas and fields.

## Four identities

Every producer identity hashes resolved setting values, input content, a hand-bumped **method revision**
and the producer's **key numerical packages**. Code bytes and the full stage environment are recorded with
each product as provenance but never decide reuse.

- **Stochastic collection** — `collection_id`, built in [`generation_plan.py`][generation] through
  `collection_id_v2` (`scenario-collection-intent/4`). Hashes the scenario spec and semantics, the
  `stochastic-collection` method revision, run-ID capacity, generation settings with the resolved seed,
  the scientific source inventory and `GENERATION_KEY_PACKAGES`.
- **Delta-change collection** — `collection_id`, built by `_publish_named` in
  [`delta_scenarios.py`][delta] (`delta-scenario-intent/4`). Hashes the intent without its `provenance`
  record: resolved settings, input file hashes, factor values, the `delta-collection` method revision and
  numerical package versions. The producer's bytes sit in `provenance` and check the retained
  `producer.py` copy.
- **WF4 simulation** — `simulation_id`, built by `_v2_simulation_id` in [`simulation_record.py`][record]
  (`simulation-intent/3`). Hashes the collection ID, revision and selected run IDs, model digest, Wflow
  revision, settings, response request, forcing preparation, the `wf4-simulation` method revision and
  `SIMULATION_KEY_PACKAGES`.
- **Indicator set** — `indicator_set_id`, built by `build_indicator_plan` in
  [`indicator_plan.py`][indicator-plan] (`indicator-request/6`, `indicator-set-identity/6`). Hashes the
  simulation ID, response identity, indicator definition (declarations and the `indicator-set` method
  revision), `INDICATOR_KEY_PACKAGES` with the return-level estimator, run groups, references, water-year
  anchor and return-level validation. One revision covers the engine and every installed set; each
  declaration's `implementation_revision` is that revision.

All digests are SHA-256 of canonical JSON (`collection-canon/1`) from
[`content_identity.py`](../../blueearth_cst/experiment/content_identity.py). The experiment name and the
collection's folder name are not identity inputs. The automatic WF3 seed derives from the method revision
(`generation-seed-material/3`), so it is stable across non-numerical edits.

Earlier schemas (`scenario-collection-intent/3`, `delta-scenario-intent/3`, `simulation-intent/2`,
`indicator-set-identity/5`) hashed code inventories and full environments. They stay readable and
consumable, but are never matched for reuse; continuing such an experiment is refused with the
`--new-experiment` advice.

## Method revisions and pins

[`method_revisions.json`](../../blueearth_cst/experiment/method_revisions.json) stores, per producer, the
method revision with the code closure it was declared for: the file list, its digest and the last record
(`bump` or `no-numerical-change`, with a reason). `method_revision()` in
[`method_revisions.py`](../../blueearth_cst/experiment/method_revisions.py) is the only way an identity
builder obtains a revision. It recomputes the closure and raises `MethodPinMismatch` naming the changed
files, so WF3 and WF4 stop at start-up on unrecorded code. A run checks only the producers it computes:
indicators-only runs check `indicator-set` alone, so a simulation-code edit does not block them.

After editing code in a closure, record the change in the same commit:

```powershell
python dev/scripts/record_method_pins.py --check
python dev/scripts/record_method_pins.py --producer wf4-simulation --no-numerical-change "log wording"
python dev/scripts/record_method_pins.py --producer wf4-simulation --bump "kinematic-wave option"
```

Bump only when produced values can change; a bump re-keys every identity of that producer in every
project. [`test_method_pins.py`](../../tests/test_method_pins.py) fails until every pin is current.

### Closures

`repository_code_inventory` hashes every file reached from declared entry paths through static
`blueearth_cst` imports, including imports inside functions and package `__init__.py` files. R and Julia
files enter only as explicit entries. CRLF is folded to LF before hashing, so a checkout's line endings
never trip a pin. The pin file is JSON and therefore in no closure.

Each producer owns its entries and an exclusion list of files outside its numerical method (logging,
archiving, configuration composition, the guard itself). Exclusions no longer affect identity; they keep
re-record churn down, so justify one by "not part of the method":

- **Stochastic:** `GENERATION_CODE_ENTRIES` and `GENERATION_INVENTORY_EXCLUDED` in
  [`generation_plan.py`][generation]. The delta producer is excluded; it has its own pin.
- **Delta:** `DELTA_CODE_ENTRIES` and `DELTA_INVENTORY_EXCLUDED` in `method_revisions.py`.
- **Simulation:** `SIMULATION_CODE_ENTRIES` and `SIMULATION_INVENTORY_EXCLUDED` in
  [`simulation_code.py`][simulation-code].
- **Indicators:** `_indicator_code_inventory` in [`indicator_plan.py`][indicator-plan] over every installed
  set.

[`test_module_import_direction.py`](../../tests/test_module_import_direction.py) keeps the console tier out
of all closures, and [`test_weather_plot_identity.py`](../../tests/test_weather_plot_identity.py) keeps
presentation files out of the generator closure. `record_method_pins.py --check` lists each closure's
added, removed and changed files. WF2 series caches carry a separate reducer hash that ignores comments; see
the [caches reference](toolbox-data-catalogs-and-caches.md#project-caches).

## What a changed identity does

- **Collection inputs** (method revision, generation settings, sources, key packages): the new
  `collection_id` finds no ready marker, so WF3 generates and publishes a new collection; a used name gets a
  `-2`, `-3` suffix ([`collection_index.py`][collection-index]). Nothing refuses. A WF4 experiment in
  project-generation mode also carries the generator method revision in its reservation request, so a
  bump refuses continuing that experiment (see [Reservations](#reservations)). Delta collections behave the
  same way.
- **Simulation inputs** (method revision, model, collection, settings, preparation, key packages):
  continuing a completed simulation is refused by `check_simulation_reuse` in
  [`simulation_reuse.py`][simulation-reuse] ("use a new experiment name"); a differing `simulation_id` is
  refused at freeze by `freeze_simulation_v2`. A differing code inventory or non-key package alone is
  provenance and does not refuse.
- **Indicator inputs** (method revision, declarations, key packages, references): a new immutable
  indicator set is published beside the old one; Wflow does not rerun.

A product continued under the same revision keeps its creator's recorded code inventory; the completing
invocation's publication record carries its own `toolbox_identity` (commit and dirty flag). An interrupted
create run resumed after a no-numerical-change edit continues in place; Snakemake may rerun steps whose
code changed. Retained products are never rewritten to match the current code.

## Reservations

[`reservation.py`][reservation] numbers experiments as `<base>_NN` families under
`experiments/_engine/allocation.json`. Writers must hold the project-execution lock for the whole
invocation; preview is advisory. Without a pin, admission continues the family's highest allocation;
`--new-experiment` allocates the next number, `--recover-reservation` completes an interrupted
publication, and indicators-only never allocates. Exact legacy names are admitted unchanged.

[`reservation_request.py`][reservation-request] captures the experiment request before producers run:
selection mode (`project-generation`, `explicit-manifest` or `delta-generation`), catalogs with their root
context, the generation request (including the generator method revision) and the simulation request.
Catalog root selection is explicit `meta.root`, the first existing `meta.roots` candidate, or the catalog
directory. The capture does not load HydroMT or validate every adapter and unit conversion; the workflow
using those inputs performs those checks.
`compare_request` refuses the first changed field and names it, suggesting the retained inputs or
`--new-experiment`. Unclassified `simulate_system`, `compute` and `build_model` fields are refused, so a new
key must be classified there; see [configuration changes](dev-configuration-changes.md).
[`reservation_context.py`][reservation-context], [`reservation_generation.py`][reservation-generation] and
[`reservation_freeze.py`][reservation-freeze] carry the admitted reservation through WF3 generation and the
WF4 freeze. Tests: [`test_reservation.py`](../../tests/test_reservation.py) and its `test_reservation_*`
siblings. [`allocate.py`](../../blueearth_cst/experiment/allocate.py) is a read-only naming compatibility
layer; it does not create experiments.

## Batching

WF4 runs selected runs in batches, one Julia session per batch, through rule 4.05 in
[`simulate_and_indicators.smk`][wf4-rules]. [`batch_sizing.py`][batch-sizing] decides at parse time:

1. `resolve_batch_plan` chooses Julia threads per batch and batches at once. Precedence is the project's
   `compute` keys, then numeric `advanced_settings.batching` values, then the small- or large-basin
   regime by active cell count. Batches at once is capped by cores per thread and by 80 % of available
   memory at the per-batch estimate.
2. `resolve_batch_size` takes the smallest applicable ceiling: parallelism, optional `batch_size_max`,
   and a disk ceiling that applies only when the whole sweep cannot fit the headroom. The per-run
   footprint is measured from WF1's retained historical forcing; an unavailable estimate removes the disk
   ceiling rather than failing. An explicit `compute.batch_size` wins unclamped, with a warning when its
   estimated peak exceeds the headroom.
3. `split_evenly` splits runs into `ceil(K/B)` consecutive batches whose sizes differ by at most one.

`batch_size_max`, an explicit `batch_size` or the disk ceiling can produce more batches than the plan's
width. Each batch rule therefore takes one unit of the `wflow_batches` resource, whose global limit the
Snakefile registers at parse time as the plan's batches at once (or a lower `--resources` value), so at
most that many batches run whatever `--cores` allows. The limit is saved with the batch decision.

Batch membership can change between attempts. [`batch_staging.py`][batch-staging] therefore stages each
finished run by run ID and reuses it on retry only when its TOML and frozen-simulation digests match.
`batch_sizing.py` is in the simulation closure, so editing it needs a `wf4-simulation` pin record.

[`scripts/calibrate_batching.py`](../../scripts/calibrate_batching.py) times thread and width settings on a
user's basin and prints `compute:` keys; it writes nothing to configuration. Its default widths are 1, 2,
then doubling up to the physical core count, less any above the batch plan's memory limit; width `w` runs
`w` batches of `--members` each.
[`dev/scripts/estimate_batch_makespan.py`](../../dev/scripts/estimate_batch_makespan.py) is the LPT
makespan estimator behind the batching design. Tests: [`test_batch_plan.py`](../../tests/test_batch_plan.py),
[`test_batch_sizing.py`](../../tests/test_batch_sizing.py),
[`test_batch_staging.py`](../../tests/test_batch_staging.py),
[`test_calibrate_batching.py`](../../tests/test_calibrate_batching.py) and
[`test_estimate_batch_makespan.py`](../../tests/test_estimate_batch_makespan.py).

## Verification

Identity and reuse changes are covered by [`test_method_pins.py`](../../tests/test_method_pins.py),
[`test_content_identity.py`](../../tests/test_content_identity.py),
[`test_scenario_collection_v2.py`](../../tests/test_scenario_collection_v2.py),
[`test_simulation_reuse.py`](../../tests/test_simulation_reuse.py),
[`test_indicator_plan_v2.py`](../../tests/test_indicator_plan_v2.py) and
[`test_delta_scenarios.py`](../../tests/test_delta_scenarios.py). Use the
[validation ladder](dev-validation.md) to select checks. A method-revision bump invalidates reuse of
retained runs in every project; state it in the change's task or commit message.

[generation]: ../../blueearth_cst/experiment/generation_plan.py
[delta]: ../../blueearth_cst/experiment/delta_scenarios.py
[record]: ../../blueearth_cst/experiment/simulation_record.py
[simulation-code]: ../../blueearth_cst/experiment/simulation_code.py
[indicator-plan]: ../../blueearth_cst/experiment/indicator_plan.py
[collection-index]: ../../blueearth_cst/experiment/collection_index.py
[simulation-reuse]: ../../blueearth_cst/experiment/simulation_reuse.py
[reservation]: ../../blueearth_cst/experiment/reservation.py
[reservation-request]: ../../blueearth_cst/experiment/reservation_request.py
[reservation-context]: ../../blueearth_cst/experiment/reservation_context.py
[reservation-generation]: ../../blueearth_cst/experiment/reservation_generation.py
[reservation-freeze]: ../../blueearth_cst/experiment/reservation_freeze.py
[wf4-rules]: ../../blueearth_cst/experiment/rules/simulate_and_indicators.smk
[batch-sizing]: ../../blueearth_cst/experiment/batch_sizing.py
[batch-staging]: ../../blueearth_cst/experiment/batch_staging.py
