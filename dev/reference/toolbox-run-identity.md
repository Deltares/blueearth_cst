# Run identity, reuse, reservations and batching

Last updated: 2026-11-14

Use this reference before changing what identifies a scenario collection, a WF4 simulation or an indicator
set, before editing a module that one of those identities hashes, and when changing experiment reservation
or WF4 batch sizing. The [retained-handoffs guide](../../docs/technical-manual/retained-handoffs.md) explains
saved products and reuse to users; the [architecture reference](toolbox-architecture.md) owns workflow
boundaries. The modules named below own exact schemas and fields.

## Three identities

- **Stochastic collection** — `collection_id`, built in [`generation_plan.py`][generation] through
  `collection_id_v2`. Hashes the scenario spec and semantics, provider revision, run-ID capacity,
  generation settings with the resolved seed, the scientific source inventory, the generator code inventory
  and the stage environment.
- **Delta-change collection** — `collection_id`, built by `_publish_named` in
  [`delta_scenarios.py`][delta]. Hashes the resolved settings without `name`, input file hashes, factor
  values, the producer file's bytes and numerical package versions.
- **WF4 simulation** — `simulation_id`, built by `_v2_simulation_id` in [`simulation_record.py`][record].
  Hashes the collection ID, revision and selected run IDs, model digest, Wflow revision, simulation code
  inventory, settings, response request, stage environment and forcing preparation.
- **Indicator set** — `indicator_set_id`, built by `build_indicator_plan` in
  [`indicator_plan.py`][indicator-plan]. Hashes the simulation ID, response identity, indicator definition
  (which contains the calculation code inventory), indicator environment, run groups, references,
  water-year anchor and return-level validation.

All digests are SHA-256 of canonical JSON (`collection-canon/1`) from
[`content_identity.py`](../../blueearth_cst/experiment/content_identity.py). The experiment name and the
collection's folder name are not identity inputs. The automatic WF3 seed derives from the same generator
code inventory (`generation_seed_material_v2`), so a code change also changes an unpinned seed.

## Code inventories

`repository_code_inventory` hashes every file reached from declared entry paths through static
`blueearth_cst` imports, including imports inside functions and package `__init__.py` files. R and Julia
files enter only as explicit entries. Each file contributes the SHA-256 of its whole content after CRLF is
folded to LF, so **a comment-only edit to any inventoried file changes the identity**. External packages are
recorded separately by `stage_environment`, which captures installed Python distributions, Conda records,
the R libraries weathergenr resolves and, for WF4, the Julia manifest.

Each caller owns its entries and an exclusion list of files judged unable to change output bytes (logging,
archiving, configuration composition). Excluded files are neither hashed nor followed. The lists live in
inventoried files, so changing one also changes the identity.

- **Generator (WF3):** `GENERATION_CODE_ENTRIES` and `GENERATION_INVENTORY_EXCLUDED` in
  [`generation_plan.py`][generation].
- **Simulation (WF4):** `SIMULATION_CODE_ENTRIES` and `SIMULATION_INVENTORY_EXCLUDED` in
  [`simulation_code.py`][simulation-code].
- **Indicator calculation:** `_indicator_code_inventory` in [`indicator_plan.py`][indicator-plan], which
  varies with the selected sets.

The delta producer is the exception: it hashes only the raw bytes of `delta_scenarios.py` itself, without
following imports and without CRLF folding.

Only the generator closure is pinned by a test
([`test_wf3_helper_boundary.py`](../../tests/test_wf3_helper_boundary.py)); a module entering or leaving it
fails that test. [`test_module_import_direction.py`](../../tests/test_module_import_direction.py) keeps the
console tier out of all three closures, and
[`test_weather_plot_identity.py`](../../tests/test_weather_plot_identity.py) keeps presentation files out of
the generator closure. The closures overlap: shared modules such as `content_identity.py` and the delta
modules sit in more than one.

### Check before editing a shared module

Compute the closures in the session runtime and look for the file you intend to edit:

```python
from pathlib import Path
from blueearth_cst.experiment.content_identity import repository_code_inventory
from blueearth_cst.experiment.generation_plan import (
    GENERATION_CODE_ENTRIES, GENERATION_INVENTORY_EXCLUDED)
from blueearth_cst.experiment.simulation_code import simulation_code_inventory
from blueearth_cst.experiment.indicator_plan import _indicator_code_inventory
from blueearth_cst.indicators.loader import available_sets

repo = Path(".").resolve()
closures = {
    "generation": repository_code_inventory(
        repo, GENERATION_CODE_ENTRIES, GENERATION_INVENTORY_EXCLUDED),
    "simulation": simulation_code_inventory(repo),
    "indicators": _indicator_code_inventory(available_sets()),
}
for name, inventory in closures.items():
    print(name, sorted(item["path"] for item in inventory))
```

`_indicator_code_inventory` is private, and passing every available set gives the widest indicator closure.
To measure a change, run the snippet on the base and on the branch and compare. A file in a closure needs
an explicit decision: accept the new identities, move the logic out of the closure, or add a justified
exclusion. Do not infer membership from a module's docstring; compute it.

## What a changed identity does

- **Collection inputs** (generator closure, generation settings, sources, environment): the new
  `collection_id` finds no ready marker, so WF3 generates and publishes a new collection; a used name gets a
  `-2`, `-3` suffix ([`collection_index.py`][collection-index]). Nothing refuses. A WF4 experiment in
  project-generation mode also carries the generator inventory in its reservation request, so continuing
  that experiment is refused (see [Reservations](#reservations)). Delta collections behave the same way.
- **Simulation inputs** (simulation closure, model, collection, settings, preparation, environment):
  continuing a completed simulation is refused by `check_simulation_reuse` in
  [`simulation_reuse.py`][simulation-reuse] ("use a new experiment name"); a differing frozen intent is
  refused at freeze by `freeze_simulation_v2`.
- **Indicator inputs** (indicator closure, declarations, environment, references): a new immutable
  indicator set is published beside the old one; Wflow does not rerun.

Indicators-only runs read the frozen simulation intent and retained responses; they do not re-observe the
simulation closure. A simulation-closure edit therefore blocks re-simulation of an existing experiment
but not new indicator sets from it. Retained products are never rewritten to match the current code.

## Reservations

[`reservation.py`][reservation] numbers experiments as `<base>_NN` families under
`experiments/_engine/allocation.json`. Writers must hold the project-execution lock for the whole
invocation; preview is advisory. Without a pin, admission continues the family's highest allocation;
`--new-experiment` allocates the next number, `--recover-reservation` completes an interrupted
publication, and indicators-only never allocates. Exact legacy names are admitted unchanged.

[`reservation_request.py`][reservation-request] captures the experiment request before producers run:
selection mode (`project-generation`, `explicit-manifest` or `delta-generation`), catalogs with their root
context, the generation request (including the generator code inventory) and the simulation request.
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

Batch membership can change between attempts. [`batch_staging.py`][batch-staging] therefore stages each
finished run by run ID and reuses it on retry only when its TOML and frozen-simulation digests match.
`batch_sizing.py` is in the simulation closure, so editing it changes simulation identities.

[`scripts/calibrate_batching.py`](../../scripts/calibrate_batching.py) times thread and width settings on a
user's basin and prints `compute:` keys; it writes nothing to configuration.
[`dev/scripts/estimate_batch_makespan.py`](../../dev/scripts/estimate_batch_makespan.py) is the LPT
makespan estimator behind the batching design. Tests: [`test_batch_plan.py`](../../tests/test_batch_plan.py),
[`test_batch_sizing.py`](../../tests/test_batch_sizing.py),
[`test_batch_staging.py`](../../tests/test_batch_staging.py),
[`test_calibrate_batching.py`](../../tests/test_calibrate_batching.py) and
[`test_estimate_batch_makespan.py`](../../tests/test_estimate_batch_makespan.py).

## Verification

Identity and reuse changes are covered by [`test_content_identity.py`](../../tests/test_content_identity.py),
[`test_scenario_collection_v2.py`](../../tests/test_scenario_collection_v2.py),
[`test_simulation_reuse.py`](../../tests/test_simulation_reuse.py),
[`test_indicator_plan_v2.py`](../../tests/test_indicator_plan_v2.py) and
[`test_delta_scenarios.py`](../../tests/test_delta_scenarios.py). Use the
[validation ladder](dev-validation.md) to select checks. A deliberate identity change invalidates reuse of
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
