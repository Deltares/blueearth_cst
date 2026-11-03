---
title: "Transferring climate-projection results"
format: html
---

`scripts/transfer_results.py` exports a completed projection analysis product as one ZIP, or imports one into
another
project. It prints a JSON report and exits zero on success. A `BundleError` prints `Refused: ...` on stderr
and returns 2. The following commands use placeholders for project roots:

```console
pixi run python scripts/transfer_results.py export --project-dir <source project> --output cmip6-results.zip
pixi run python scripts/transfer_results.py import cmip6-results.zip --project-dir <destination project>
```

## Bundle contents and export checks

The bundle holds the whole product (including `raw/`, `scalar/` and `_engine/`), the exact configuration
snapshots that produced it, the basin `region.geojson`, and a `manifest.json` listing every file's size and
SHA-256. It needs no source-machine paths, `_shared/` execution inputs, locks or `.snakemake/` state.
Source paths inside copied configuration are history, listed under `historical_path_references`.
Use `--ensemble <name>` when the project has no captured projection analysis configuration naming the product.

Export refuses an `incomplete` product, missing required files, a `region.geojson` whose polygon differs
from the one recorded in `summary/provenance.json`, and provenance that matches no retained
configuration snapshot. Files whose bytes no longer match the production record are exported with
`producer: null`. Products without production records export as `legacy-unattributed`; their
configuration snapshot is the one whose digests match `summary/provenance.json`.

The manifest schema is `cst-result-bundle/1`. Required product files include `report.md`,
`summary/provenance.json`, `summary/composition.csv` and `_engine/analysis_record.json`.
Linked export paths are rejected. Export writes a new ZIP rather than replacing an existing destination.

## Destination checks and import

Import checks the whole bundle before writing: member names, sizes, checksums, the manifest's identity,
schema version and required files. It then compares the destination:

| Destination check | On mismatch |
|---|---|
| Basin polygon in `data/spatial/geoms/` | Rejected, even with an equal bounding box; absent is `unverified` |
| projection analysis ensemble, models, scenarios, members, variables, windows | Rejected unless `--accept-differences` |
| Existing `data/climate/projections/<ensemble>/` | Rejected without changing existing results; the same bundle is `already_imported` |

Settings are compared with `--config` when given, else with the destination's captured projection analysis
configuration. Accepted differences are recorded in the import record.

`--accept-differences` applies only to projection analysis scientific settings, not geometry, bundle
integrity or an existing
different product. There is no merge/overwrite flag for that product. Keep it and choose a separate project
when the destination cannot accept the bundle.

The product is staged in a hidden `.<ensemble>.import-*.staged` sibling and appears only after
verification, with one rename. A staged folder left by an interruption is never a result; the next
import removes it. Import adds `_engine/imported/<bundle id>/` with the manifest, `import_record.json`,
the configuration snapshots and region. It does not write the destination's `config/` archives,
`data/spatial/`, invocation records or logs; it only holds the project's execution lock under
`config/_engine/archive-transactions/`.

## Using imported results {#post-import-limits}

Imported results are for reading and interpretation. The runner has no record of producing them, so a
later projection analysis run in the destination may recompute them, and projection analysis does not need to
be enabled to read them.
The bundle records toolbox commit, dirty state and lock hashes as provenance; they do not recreate the
source environment.

The transfer implementation is `blueearth_cst/shared/result_bundle.py`; command-line options are defined in
`scripts/transfer_results.py`. `tests/test_result_bundle.py` covers integrity, destination comparison,
publication and preservation/refusal cases. [Product
attribution](execution-and-provenance.md#wf2-product-attribution)
explains the production record carried in a bundle.
