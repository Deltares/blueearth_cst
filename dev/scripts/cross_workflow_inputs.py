"""The wf1 leaves WF2/WF3 declare and Snakemake will not satisfy on its own.

A CROSS-WORKFLOW LEAF is a file some rule in WF2 or WF3 declares as an input
while no rule in that workflow declares it as an output. Snakemake cannot
produce it, so anything driving WF2/WF3 in isolation — a dry-run test, the
layout scaffolder — has to put it on disk first.

**Why this is one definition rather than three.** It was three, and they drifted
independently. R9 P4's rule 3.01c `write_model_reference` is the first WF3 rule
ever to declare model FILES as inputs; before it, WF3 reached the model only
through `params` and the DAG could not see the dependency. Two of the three
copies were updated for it and one was not, so `test_guard_invalidation`'s gate
2c(iii) went red in a way that read as a guard defect (R9 P5 F3). The third copy,
`scaffold_project_tree.py`, was never updated at all and had also fallen two
milestones behind on the model root, so WF3 contributed 0 of its 95 declared
outputs while the tool still exited 0.

**A shared list still goes stale — it just does so in one place.** What stops it
is `tests/test_cross_workflow_inputs.py`, which proves against the real DAG that
`LEAVES` is COMPLETE (staging exactly it lets WF2 and WF3 dry-runs resolve) and
MINIMAL (drop any one and a dry-run fails). A rule declaring a new leaf turns
that test red immediately, which is the escape above.

**Extras are not leaves.** Both test fixtures also stage files the DAG does not
require, for reasons that are real but separate — see `EXTRA_*` below. They are
passed explicitly so that "required by Snakemake" and "wanted by this caller"
stay legible; folding them into `LEAVES` is how the vestigial ones survived.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

#: WF3 rule 3.00b `check_project_consistency` takes this as a mandatory
#: `ancient()` input; its absence is a rule-level MissingInputException.
LEAF_WF1_SNAPSHOT = "config/runs/snake_config_build_model.yml"

#: WF3 rule 3.01c `write_model_reference` (R9 P4), the first WF3 rule to declare
#: model files as inputs. Both are `ancient()`: the reference must not re-derive
#: because the model was rebuilt, only because the config moved.
LEAF_MODEL_TOML = "models/hydrology/wflow/wflow_sbm.toml"
LEAF_MODEL_READY = "models/hydrology/wflow/.outputs_configured"

#: The full set. Order is stable so failure messages read the same way twice.
LEAVES: tuple[str, ...] = (LEAF_WF1_SNAPSHOT, LEAF_MODEL_TOML, LEAF_MODEL_READY)

# --- Deliberate NON-leaves -------------------------------------------------
# Staged by some callers, required by no DAG. Each is here with its reason so
# the next reader does not have to decide whether it is drift.

#: NOT a declared input anywhere: WF3's drift guard reads it as a `params` path
#: and existence-checks it in the script, because the projections overlay is
#: optional and must not be force-required. `test_guard_invalidation` stages it
#: because its assertions are about the guard's COMPARISON, not the DAG.
EXTRA_WF2_SNAPSHOT = "config/runs/snake_config_analyze_projections.yml"

#: NOT read by either downstream workflow. R07 B1 retired the extraction's
#: `ancient(region.geojson)` input and ADR 0003 gave WF2 and WF3 their own
#: `delineate_region`, each declaring `data/spatial/geoms/region.geojson` as an
#: OUTPUT. Staged by the two test fixtures only to keep the scratch project
#: looking like a completed wf1 run.
EXTRA_REGION = "models/hydrology/wflow/staticgeoms/region.geojson"

#: Enough of a wflow TOML for rule 3.01c to read `input.path_static`.
MINIMAL_WFLOW_TOML = '[input]\npath_static = "staticmaps.nc"\n'

#: Minimal valid polygon, for callers that stage `EXTRA_REGION`.
MINIMAL_REGION_GEOJSON = """{
  "type": "FeatureCollection",
  "features": [{
    "type": "Feature", "properties": {"value": 1},
    "geometry": {"type": "Polygon", "coordinates": [[
      [11.3, -1.05], [13.6, -1.05], [13.6, 0.9], [11.3, 0.9], [11.3, -1.05]]]}
  }]
}
"""

#: Paths whose content must be the run's own config, so a snapshot-comparing
#: guard sees identical comparands and passes by construction.
_CONFIG_SNAPSHOTS = frozenset({LEAF_WF1_SNAPSHOT, EXTRA_WF2_SNAPSHOT})


def content_for(rel: str, config_text: str) -> str:
    """Return the file content to stage at project-relative path ``rel``."""
    if rel in _CONFIG_SNAPSHOTS:
        return config_text
    if rel.endswith("wflow_sbm.toml"):
        return MINIMAL_WFLOW_TOML
    if rel.endswith("region.geojson"):
        return MINIMAL_REGION_GEOJSON
    return ""


def stage(
    project_dir: Path,
    config_text: str,
    extras: Sequence[str] = (),
    leaves: Iterable[str] | None = None,
) -> tuple[Path, ...]:
    """Materialize the cross-workflow leaves (plus ``extras``) under ``project_dir``.

    Args:
        project_dir: Scratch project root. Created if absent.
        config_text: Serialized config to write at any snapshot path. Serialize
            it from the SAME parsed config the run consumes, or a snapshot-
            comparing guard will fail on a difference the caller invented.
        extras: Deliberate non-leaves — pass the ``EXTRA_*`` constants.
        leaves: Override the leaf set. For the minimality proof only; callers
            staging a project should leave it ``None``.

    Returns:
        The staged paths, in the order written.
    """
    staged: list[Path] = []
    for rel in (*(LEAVES if leaves is None else leaves), *extras):
        target = Path(project_dir) / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content_for(rel, config_text), encoding="utf-8")
        staged.append(target)
    return tuple(staged)
