# Decision records (ADRs)

Architecture / method / tooling decision records for blueearth_cst. One ADR per
subject area; revise in place as a design evolves, supersede only on a genuine
reversal. Format: `dev/decisions/NNNN-<slug>.md` (see the `design-document`
skill's decision-record reference).

| # | Title | Status | Date |
|---|---|---|---|
| [0001](0001-restore-wflow-constant-parameters.md) | Reconcile the dropped Wflow constant parameters via evidence-gated CSDMS restoration | accepted | 2026-07-21 |
| [0002](0002-revive-subcatchment-climate-plots.md) | Revive the subcatchment climate plots from the wflow forcing input | superseded by 0006 | 2026-07-21 |
| [0003](0003-one-shared-region-artifact.md) | Spatial artifacts delineated once per project, shared across workflows | accepted §1–12 | 2026-08-06 |
| [0004](0004-order-model-readers-on-a-terminal-sentinel.md) | Order model-root readers on a terminal build sentinel, not on a declared output | accepted | 2026-08-05 |
| [0005](0005-adopt-ruff-format-in-two-stages.md) | Adopt `ruff format`, in two stages split on the Snakemake code rerun trigger | accepted (both stages landed) | 2026-08-09 |
| [0006](0006-retire-subcatchment-climate-plots.md) | Retire the subcatchment climate plots; the canonical climate figure set answers this | accepted | 2026-08-09 |
| [0007](0007-draw-basin-area-from-the-spatial-foundation.md) | Draw basin_area from the spatial foundation, not the model | accepted | 2026-08-09 |
| [0008](0008-ship-blueearth-cst-unpackaged.md) | Ship `blueearth_cst` unpackaged; `pyproject.toml` stays tool-config-only | accepted | 2026-08-17 |

## Retired numbers

_none._
